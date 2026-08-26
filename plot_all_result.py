import argparse
import os
import re
from typing import Dict, List, Tuple
import numpy as np

import matplotlib.pyplot as plt

# ── Font: SimSun (宋体) for Chinese, Times New Roman for English ──
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["SimSun", "SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# ── Experiment matrix ──────────────────────────────────────────────
MODELS = ["distilbert", "roberta", "llama"]
DATASETS = ["20news", "squad"]
DISTRIBUTIONS = ["iid", "niid"]
# DATASETS = ["20news"]
# DISTRIBUTIONS = ["iid"]
METHODS = ["raw", "helora", "ffthm", "pq", "topk"]

METHOD_MARKERS = {
    "raw": "o",
    "HeLoRA": "s",
    "helora": "s",
    "FFTHM": "^",
    "ffthm": "^",
    "pq": "D",
    "topk": "v",
    "topk_ab": "P",
}

METHOD_DISPLAY = {
    "raw": "FedIT",
    "helora": "HeLoRA-Pad",
    "ffthm": "FFTHM",
    "pq": "FedPAQ_LoRA",
    "topk": "FLASC",
}

MODEL_DISPLAY = {
    "distilbert": "DistilBERT",
    "roberta": "RoBERTa",
    "llama": "LLaMA",
}

DATASETS_DISPLAY = {
    "20news": "20Newsgroups",
    "squad": "SQuAD",
}

# ── Regex patterns ─────────────────────────────────────────────────
ROUND_RE = re.compile(r"\bROUND\s*:\s*(\d+)\b", re.IGNORECASE)
EVAL_ACC_RE = re.compile(r"[\"']eval_acc[\"']\s*:\s*([0-9eE+\-\.]+)")
EVAL_F1_RE = re.compile(r"[\"']eval_f1[\"']\s*:\s*([0-9eE+\-\.]+)")
SECONDS_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*s\b", re.IGNORECASE)


# ── Helpers ────────────────────────────────────────────────────────
def infer_method_from_filename(path: str) -> str:
    """Extract method name from the last underscore-delimited segment of the stem."""
    base = os.path.basename(path)
    stem, _ = os.path.splitext(base)
    if "_" in stem:
        return stem.split("_")[-1]
    return stem


def get_marker(method: str) -> str:
    return METHOD_MARKERS.get(method.lower(), "o")


def get_display_name(method: str) -> str:
    return METHOD_DISPLAY.get(method.lower(), method)


def parse_log(path: str, dataset: str) -> Tuple[str, List[Tuple[int, float]], Dict[int, float]]:
    """Parse one .out log file.

    Returns
    -------
    method : str
    acc_by_round : list of (round, accuracy)
    cumulative_time_by_round : dict round → cumulative training seconds
    """
    method = infer_method_from_filename(path)
    eval_re = EVAL_F1_RE if dataset == "squad" else EVAL_ACC_RE

    acc_by_round: List[Tuple[int, float]] = []
    cumulative_time_by_round: Dict[int, float] = {}

    current_round = -1
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            m_round = ROUND_RE.search(line)
            if m_round:
                current_round = int(m_round.group(1))
                continue

            # Accuracy line  (Trainer eval dict)
            m_acc = eval_re.search(line)
            if m_acc and current_round >= 0:
                acc = float(m_acc.group(1))
                acc_by_round.append((current_round, acc))
                continue

            # Cumulative-time line  (Chinese / English variants)
            m_seconds = SECONDS_RE.search(line)
            if not m_seconds or current_round < 0:
                continue
            sec = float(m_seconds.group(1))
            lower = line.lower()
            if ("累计训练时间" in line) or ("cumulative" in lower) or ("total training time" in lower):
                cumulative_time_by_round[current_round] = sec

    # Deduplicate by round (some logs print the eval dict twice)
    dedup: Dict[int, float] = {}
    for rnd, acc in acc_by_round:
        dedup[rnd] = acc
    sorted_acc = sorted(dedup.items(), key=lambda x: x[0])

    return method, sorted_acc, cumulative_time_by_round



# ── Main ───────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot 4 figures (3 subplots each): cumulative time vs accuracy "
                    "for all model×dataset×distribution×method combinations."
    )
    parser.add_argument("--dpi", type=int, default=200, help="Figure DPI")
    args = parser.parse_args()

    # ── 1. Auto-generate file list & parse ─────────────────────────
    # Nested dict: results[dataset][distribution][model][method] = (times, accs)
    ResultsType = Dict[str, Dict[str, Dict[str, Dict[str, Tuple[List[float], List[float]]]]]]
    results: ResultsType = {ds: {dist: {m: {} for m in MODELS} for dist in DISTRIBUTIONS} for ds in DATASETS}

    missing_files: List[str] = []
    skipped_no_time: List[str] = []

    for ds in DATASETS:
        for dist in DISTRIBUTIONS:
            for model in MODELS:
                for method in METHODS:
                    path = os.path.join("result", f"{model}_{ds}_{dist}_{method}.out")
                    if not os.path.isfile(path):
                        missing_files.append(path)
                        continue

                    method_name, acc_pairs, cum_time_map = parse_log(path, ds)
                    if not acc_pairs:
                        skipped_no_time.append(path)
                        continue

                    rounds = [r for r, _ in acc_pairs]
                    accs = [a for _, a in acc_pairs]
                    times = [cum_time_map.get(r, float("nan")) for r in rounds]

                    valid = [(t, a) for t, a in zip(times, accs) if not np.isnan(t)]
                    if not valid:
                        skipped_no_time.append(path)
                        continue

                    valid_times = [t for t, _ in valid]
                    valid_accs = [a for _, a in valid]
                    results[ds][dist][model][method] = (valid_times, valid_accs)

    if missing_files:
        print(f"⚠  {len(missing_files)} missing files (skipped):")
        for p in missing_files:
            print(f"   {p}")
    if skipped_no_time:
        print(f"⚠  {len(skipped_no_time)} files with no usable time/acc data (skipped):")
        for p in skipped_no_time:
            print(f"   {p}")

    # ── Manual x-axis upper-bound overrides (model, dataset, distribution) → seconds ─
    X_LIMIT_OVERRIDE = {
        ("llama", "squad", "niid"): 8000,
        ("roberta", "squad", "niid"): 1500,
    }

    # ── 2. Create 4 figures ────────────────────────────────────────
    metric_label = {  # Y-axis label per dataset
        "20news": "准确率",
        "squad": "F1值",
    }

    for ds in DATASETS:
        for dist in DISTRIBUTIONS:
            fig, axes = plt.subplots(1, 3, figsize=(8, 2.8), dpi=args.dpi)
            fig.suptitle(f"{DATASETS_DISPLAY.get(ds, ds)}  —  {dist.upper()}", fontsize=10, fontweight="bold", y=0.99,
                         fontfamily="Times New Roman")

            for ax_idx, model in enumerate(MODELS):
                ax = axes[ax_idx]
                model_data = results[ds][dist][model]
                has_data = False

                # ── Time bound: shortest method's total time ────────
                t_bound = X_LIMIT_OVERRIDE.get((model, ds, dist))
                if t_bound is None:
                    final_times = [model_data[m][0][-1] for m in METHODS
                                   if m in model_data and model_data[m]]
                    t_bound = min(final_times) if final_times else None

                for method in METHODS:
                    if method not in model_data or not model_data[method]:
                        continue
                    times, accs = model_data[method]

                    # ── Clip data to t_bound: keep points ≤ bound ──
                    if t_bound is not None:
                        clip_idx = next((i for i, t in enumerate(times) if t > t_bound), len(times))
                        times = times[:clip_idx]
                        accs = accs[:clip_idx]

                    if not times:
                        continue

                    has_data = True
                    # Thin out markers to avoid overcrowding (≈10 markers per curve)
                    step = max(1, len(times) // 10)
                    ax.plot(
                        times,
                        accs,
                        marker=get_marker(method),
                        markersize=3,
                        markevery=step,
                        linewidth=1.5,
                        label=get_display_name(method),
                    )

                if has_data:
                    if t_bound is not None:
                        ax.set_xlim(0, t_bound * 1.02)
                    ax.set_xlabel("累计训练时间（秒）")
                    ax.set_ylabel(metric_label.get(ds, "Accuracy"))
                    ax.grid(True, linestyle="--", alpha=0.4)
                else:
                    ax.text(0.5, 0.5, "No data",
                            ha="center", va="center", transform=ax.transAxes,
                            fontsize=12, color="gray")
                    ax.set_title(model, fontsize=11, fontfamily="Times New Roman")
                    continue

                ax.set_title(MODEL_DISPLAY.get(model, model), fontsize=11,
                              fontfamily="Times New Roman")

            # Single figure-level legend (all subplots share the same methods)
            for ax in axes:
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    fig.legend(handles, labels, fontsize=8, loc="upper right",
                               ncol=len(METHODS), bbox_to_anchor=(0.8, 0.95),
                               prop={"family": "Times New Roman"})
                    break

            plt.subplots_adjust(left=0.069, right=0.99, bottom=0.166, top=0.766, wspace=0.25)

            print(f"Showing: {ds} — {dist.upper()}  (close window to see next figure)")
            plt.show()


if __name__ == "__main__":
    main()
