import argparse
import glob
import os
import re
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


# Manually set your input files/globs/directories here.
MANUAL_INPUTS = [
    "result/roberta_20news_iid_raw.out",
    "result/roberta_20news_iid_helora.out",
    "result/roberta_20news_iid_ffthm.out",
    "result/roberta_20news_iid_pq.out",
    "result/roberta_20news_iid_topk.out",
]

ROUND_RE = re.compile(r"\bROUND\s*:\s*(\d+)\b", re.IGNORECASE)
METHOD_RE = re.compile(r"method\s*=\s*['\"]?([A-Za-z0-9_\-]+)['\"]?")
EVAL_ACC_RE = re.compile(r"[\"']eval_acc[\"']\s*:\s*([0-9eE+\-\.]+)")
EVAL_RUNTIME_RE = re.compile(r"[\"']eval_runtime[\"']\s*:\s*([0-9eE+\-\.]+)")
SECONDS_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*s\b", re.IGNORECASE)


def infer_method_from_filename(path: str) -> str:
    base = os.path.basename(path)
    stem, _ = os.path.splitext(base)
    if "_" in stem:
        return stem.split("_")[-1]
    return stem


def collect_files(paths: List[str]) -> List[str]:
    all_files: List[str] = []
    for p in paths:
        if os.path.isdir(p):
            all_files.extend(glob.glob(os.path.join(p, "*.out")))
            all_files.extend(glob.glob(os.path.join(p, "*.log")))
            continue
        matched = glob.glob(p)
        if matched:
            all_files.extend([m for m in matched if os.path.isfile(m)])
        elif os.path.isfile(p):
            all_files.append(p)
    # Keep order while de-duplicating
    seen = set()
    ordered_unique = []
    for f in all_files:
        ab = os.path.abspath(f)
        if ab not in seen:
            seen.add(ab)
            ordered_unique.append(ab)
    return ordered_unique


def parse_log(path: str) -> Tuple[str, List[Tuple[int, float]], Dict[int, float], Dict[int, float], Dict[int, float]]:
    method = infer_method_from_filename(path)
    acc_by_round: List[Tuple[int, float]] = []
    eval_runtime_by_round: Dict[int, float] = {}
    round_time_by_round: Dict[int, float] = {}
    cumulative_time_by_round: Dict[int, float] = {}

    current_round = -1
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            m_method = METHOD_RE.search(line)
            if m_method:
                method = m_method.group(1)

            m_round = ROUND_RE.search(line)
            if m_round:
                current_round = int(m_round.group(1))
                continue

            # Parse accuracy + eval_runtime from Trainer eval dict line.
            m_acc = EVAL_ACC_RE.search(line)
            if m_acc and current_round >= 0:
                acc = float(m_acc.group(1))
                acc_by_round.append((current_round, acc))
                m_eval_runtime = EVAL_RUNTIME_RE.search(line)
                if m_eval_runtime:
                    eval_runtime_by_round[current_round] = float(m_eval_runtime.group(1))
                continue

            # Parse optional timing lines from stdout.
            m_seconds = SECONDS_RE.search(line)
            if not m_seconds or current_round < 0:
                continue
            sec = float(m_seconds.group(1))
            lower = line.lower()
            if ("累计训练时间" in line) or ("cumulative" in lower) or ("total training time" in lower):
                cumulative_time_by_round[current_round] = sec
            elif ("本轮总耗时" in line) or ("round total time" in lower) or ("max_total_time" in lower):
                round_time_by_round[current_round] = sec

    # Keep only unique round entries (some logs may print duplicated eval dict lines)
    dedup: Dict[int, float] = {}
    for rnd, acc in acc_by_round:
        dedup[rnd] = acc
    sorted_acc = sorted(dedup.items(), key=lambda x: x[0])

    return method, sorted_acc, eval_runtime_by_round, round_time_by_round, cumulative_time_by_round


def build_time_axis(
    rounds: List[int],
    eval_runtime_by_round: Dict[int, float],
    round_time_by_round: Dict[int, float],
    cumulative_time_by_round: Dict[int, float],
) -> List[float]:
    # Priority 1: direct cumulative training time from logs.
    if cumulative_time_by_round:
        return [cumulative_time_by_round.get(r, float("nan")) for r in rounds]

    # Priority 2: per-round total time from logs, accumulated.
    if round_time_by_round:
        total = 0.0
        axis: List[float] = []
        for r in rounds:
            total += round_time_by_round.get(r, 0.0)
            axis.append(total)
        return axis

    # Priority 3: fallback to cumulative eval_runtime (always available in Trainer eval dict).
    total = 0.0
    axis = []
    for r in rounds:
        total += eval_runtime_by_round.get(r, 0.0)
        axis.append(total)
    return axis


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read nohup out/log files and plot method comparison: cumulative time vs accuracy."
    )
    parser.add_argument("--output", default="method_time_accuracy.png", help="Output figure path")
    parser.add_argument("--title", default="Method Comparison: Cumulative Time vs Accuracy", help="Figure title")
    parser.add_argument("--dpi", type=int, default=200, help="Figure DPI")
    args = parser.parse_args()

    files = collect_files(MANUAL_INPUTS)
    if not files:
        raise FileNotFoundError("No out/log files found. Check input paths or globs.")

    plt.figure(figsize=(10, 6))
    plotted = 0
    for path in files:
        method, acc_pairs, eval_runtime_map, round_time_map, cumulative_time_map = parse_log(path)
        if not acc_pairs:
            continue
        rounds = [r for r, _ in acc_pairs]
        accs = [a for _, a in acc_pairs]
        times = build_time_axis(rounds, eval_runtime_map, round_time_map, cumulative_time_map)
        plt.plot(times, accs, marker="o", markersize=3, linewidth=1.5, label=method)
        plotted += 1

    if plotted == 0:
        raise RuntimeError("No usable eval_acc entries found in provided logs.")

    plt.xlabel("Cumulative Time (s)")
    plt.ylabel("Accuracy (eval_acc)")
    plt.title(args.title)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()
    # plt.savefig(args.output, dpi=args.dpi)
    # print(f"Saved figure to: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
