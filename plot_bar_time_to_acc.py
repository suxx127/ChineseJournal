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
METHODS = ["raw", "helora", "ffthm", "pq", "topk"]

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

# Bar colours per method  (consistent with line-marker convention)
METHOD_COLORS = {
    "raw": "#1f77b4",     # blue
    "helora": "#ff7f0e",  # orange
    "ffthm": "#2ca02c",   # green
    "pq": "#9467bd",      # purple
    "topk": "#d62728",    # red
}

# Hatch patterns for black & white distinguishability
METHOD_HATCHES = {
    "raw": "",
    "helora": "//",
    "ffthm": "\\\\",
    "pq": "xx",
    "topk": "oo",
}

# ── Regex patterns ─────────────────────────────────────────────────
ROUND_RE = re.compile(r"\bROUND\s*:\s*(\d+)\b", re.IGNORECASE)
EVAL_ACC_RE = re.compile(r"[\"']eval_acc[\"']\s*:\s*([0-9eE+\-\.]+)")
EVAL_F1_RE = re.compile(r"[\"']eval_f1[\"']\s*:\s*([0-9eE+\-\.]+)")
SECONDS_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*s\b", re.IGNORECASE)


# ── Helpers ────────────────────────────────────────────────────────
def infer_method_from_filename(path: str) -> str:
    base = os.path.basename(path)
    stem, _ = os.path.splitext(base)
    if "_" in stem:
        return stem.split("_")[-1]
    return stem


def get_display_name(method: str) -> str:
    return METHOD_DISPLAY.get(method.lower(), method)


def get_color(method: str) -> str:
    return METHOD_COLORS.get(method.lower(), "#333333")


def get_hatch(method: str) -> str:
    return METHOD_HATCHES.get(method.lower(), "")


def parse_log(path: str, dataset: str) -> Tuple[str, List[Tuple[int, float]], Dict[int, float]]:
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

            m_acc = eval_re.search(line)
            if m_acc and current_round >= 0:
                acc = float(m_acc.group(1))
                acc_by_round.append((current_round, acc))
                continue

            m_seconds = SECONDS_RE.search(line)
            if not m_seconds or current_round < 0:
                continue
            sec = float(m_seconds.group(1))
            lower = line.lower()
            if ("累计训练时间" in line) or ("cumulative" in lower) or ("total training time" in lower):
                cumulative_time_by_round[current_round] = sec

    dedup: Dict[int, float] = {}
    for rnd, acc in acc_by_round:
        dedup[rnd] = acc
    sorted_acc = sorted(dedup.items(), key=lambda x: x[0])

    return method, sorted_acc, cumulative_time_by_round


def compute_time_to_target(times: List[float], accs: List[float],
                           target_acc: float) -> float | None:
    """Return the first time where accuracy >= target_acc (linear interpolation).

    Returns None if the curve never reaches the target.
    """
    for i in range(len(accs)):
        if accs[i] >= target_acc:
            if i == 0:
                return times[0]
            t0, t1 = times[i - 1], times[i]
            a0, a1 = accs[i - 1], accs[i]
            if a1 == a0:
                return t0
            return t0 + (t1 - t0) * (target_acc - a0) / (a1 - a0)
    return None


# ── Main ───────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bar charts: time to reach a common target accuracy for each method."
    )
    parser.add_argument("--dpi", type=int, default=200, help="Figure DPI")
    args = parser.parse_args()

    # ── 1. Parse all logs ──────────────────────────────────────────
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
    metric_label = {
        "20news": "训练时间（秒）",
        "squad": "训练时间（秒）",
    }

    for ds in DATASETS:
        for dist in DISTRIBUTIONS:
            fig, axes = plt.subplots(1, 3, figsize=(8, 2.8), dpi=args.dpi)
            fig.suptitle(f"{DATASETS_DISPLAY.get(ds, ds)}  —  {dist.upper()}", fontsize=10, fontweight="bold", y=0.99,
                         fontfamily="Times New Roman")

            for ax_idx, model in enumerate(MODELS):
                ax = axes[ax_idx]
                model_data = results[ds][dist][model]

                # Only keep methods that have data
                available = {m: d for m, d in model_data.items() if d}
                if len(available) < 2:
                    ax.text(0.5, 0.5, "Not enough data",
                            ha="center", va="center", transform=ax.transAxes,
                            fontsize=12, color="gray")
                    ax.set_title(MODEL_DISPLAY.get(model, model), fontsize=11,
                                  fontfamily="Times New Roman")
                    continue

                # ── Target accuracy ────────────────────────────────
                # Use best final accuracy among methods as reference;
                # drop methods whose final accuracy lags too far behind.
                final_accs = {m: d[1][-1] for m, d in available.items()}
                best_final = max(final_accs.values())

                # Filter: keep methods within 15% of the best final accuracy
                threshold = best_final * 0.85
                qualified = {m: d for m, d in available.items()
                             if final_accs[m] >= threshold}

                # If filtering left too few methods, relax the threshold
                if len(qualified) < 2:
                    threshold = best_final * 0.75
                    qualified = {m: d for m, d in available.items()
                                 if final_accs[m] >= threshold}

                # Target = minimum final accuracy among qualified methods
                target_acc = min(d[1][-1] for d in qualified.values())

                # ── Compute time-to-target for each method ─────────
                bar_labels: List[str] = []
                bar_times: List[float] = []
                bar_colors: List[str] = []
                bar_hatches: List[str] = []
                bar_display: List[str] = []
                excluded: List[str] = []  # Methods filtered out

                for method in METHODS:
                    if method not in available:
                        continue
                    if method not in qualified:
                        excluded.append(get_display_name(method))
                        continue
                    times, accs = qualified[method]
                    t = compute_time_to_target(times, accs, target_acc)
                    if t is None:
                        excluded.append(get_display_name(method))
                        continue
                    bar_labels.append(method)
                    bar_times.append(t)
                    bar_colors.append(get_color(method))
                    bar_hatches.append(get_hatch(method))
                    bar_display.append(get_display_name(method))

                # Annotate excluded methods in title
                extra = ""
                if excluded:
                    extra = f"  [excl: {', '.join(excluded)}]"

                if len(bar_times) < 2:
                    ax.text(0.5, 0.5, "Not enough data",
                            ha="center", va="center", transform=ax.transAxes,
                            fontsize=12, color="gray")
                    ax.set_title(MODEL_DISPLAY.get(model, model), fontsize=11,
                                  fontfamily="Times New Roman")
                    continue

                x = np.arange(len(bar_labels))
                bars = ax.bar(x, bar_times, color=bar_colors, width=0.55,
                              edgecolor="black", linewidth=0.8)

                # Apply hatch patterns for B&W distinguishability
                for bar, hatch in zip(bars, bar_hatches):
                    bar.set_hatch(hatch)

                # Annotate time values on top of bars
                for bar, t_val in zip(bars, bar_times):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                            f"{t_val:.1f}", ha="center", va="bottom", fontsize=7)

                ax.set_xticks(x)
                ax.set_xticklabels(bar_display, fontsize=8, rotation=20,
                                   fontfamily="Times New Roman")
                ax.set_xlabel("")
                ax.set_ylabel(metric_label.get(ds, "训练时间（秒）"))
                ax.set_title(MODEL_DISPLAY.get(model, model), fontsize=11,
                              fontfamily="Times New Roman")

                # Y-axis upper bound
                auto_max = max(bar_times) * 1.15
                y_override = X_LIMIT_OVERRIDE.get((model, ds, dist))
                if y_override is not None:
                    ax.set_ylim(0, max(y_override, auto_max))
                else:
                    ax.set_ylim(0, auto_max)

                ax.grid(True, linestyle="--", alpha=0.4)

            plt.subplots_adjust(left=0.069, right=0.99, bottom=0.16, top=0.86, wspace=0.25)
            print(f"Showing: {ds} — {dist.upper()}  (close window to see next figure)")
            plt.show()


if __name__ == "__main__":
    main()
