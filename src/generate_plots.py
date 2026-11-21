"""Generate publication-quality plots for HDCA evaluation and case studies."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from src.config import SEED, set_seed

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"

PALETTE = {
    "Solo": "#64748b",
    "UMV": "#d97706",
    "HDCA": "#0f766e",
}


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 220,
    })


def plot_single_metric(metric_name: str, values: list, filename: str, ylabel: str) -> None:
    methods = ["Solo Baseline", "UMV", "Proposed HDCA"]
    colors = [PALETTE["Solo"], PALETTE["UMV"], PALETTE["HDCA"]]
    fig, ax = plt.subplots(figsize=(6.4, 3.6), constrained_layout=True)
    bars = ax.bar(methods, values, color=colors, width=0.5, edgecolor="#1e293b", linewidth=0.7)
    ax.set_ylim(70, 100)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{metric_name} Comparison Across Evaluation Baselines")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            val + 0.7,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="#0f172a",
        )
    fig.savefig(RESULTS_DIR / filename, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_multi_metric() -> None:
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    solo = [81.2, 79.5, 82.1, 80.8]
    umv = [85.4, 82.1, 80.5, 81.3]
    hdca = [92.5, 91.8, 90.4, 91.1]

    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(7.2, 4.0), constrained_layout=True)
    b1 = ax.bar(x - width, solo, width, label="Solo Baseline", color=PALETTE["Solo"], edgecolor="#1e293b", linewidth=0.6)
    b2 = ax.bar(x, umv, width, label="UMV", color=PALETTE["UMV"], edgecolor="#1e293b", linewidth=0.6)
    b3 = ax.bar(x + width, hdca, width, label="Proposed HDCA", color=PALETTE["HDCA"], edgecolor="#1e293b", linewidth=0.6)

    ax.set_ylim(70, 100)
    ax.set_ylabel("Score (%)")
    ax.set_title("Multi-Metric Performance Comparison: Solo vs. UMV vs. HDCA")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)

    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                h + 0.5,
                f"{h:.1f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
            )

    fig.savefig(RESULTS_DIR / "multi_metric_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_exam_shift_trajectory() -> None:
    agents = ["Agent A\n(Cautious)", "Agent B\n(Neutral)", "Agent C\n(Lenient)"]
    r1 = [1.0, 3.0, 6.0]
    r2 = [1.0, 1.0, 5.0]
    x = np.arange(len(agents))
    width = 0.3

    fig, ax = plt.subplots(figsize=(6.8, 3.8), constrained_layout=True)
    b1 = ax.bar(x - width / 2, r1, width, label="Round 1 (Private)", color="#94a3b8", edgecolor="#1e293b", linewidth=0.6)
    b2 = ax.bar(x + width / 2, r2, width, label="Round 2 (Post-Critique)", color=PALETTE["HDCA"], edgecolor="#1e293b", linewidth=0.6)

    ax.axhline(2.0, color=PALETTE["Solo"], linestyle="--", linewidth=1.4, label="Solo Agent Baseline (2.00/10)")
    ax.axhline(2.33, color=PALETTE["UMV"], linestyle="-.", linewidth=1.6, label="HDCA Committee Avg (2.33/10)")

    ax.set_ylim(0, 7.5)
    ax.set_ylabel("Assigned Score (out of 10)")
    ax.set_title("Exam Grading Case: Round 1 vs. Round 2 Agent Score Shifts (66.7% Shifted)")
    ax.set_xticks(x)
    ax.set_xticklabels(agents)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                h + 0.15,
                f"{h:.1f}",
                ha="center",
                va="bottom",
                fontsize=9.5,
                fontweight="bold",
            )

    fig.savefig(RESULTS_DIR / "exam_score_shift_trajectory.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    set_seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()

    plot_single_metric("Accuracy", [81.2, 85.4, 92.5], "accuracy_comparison.png", "Accuracy (%)")
    plot_single_metric("Precision", [79.5, 82.1, 91.8], "precision_comparison.png", "Precision (%)")
    plot_single_metric("Recall", [82.1, 80.5, 90.4], "recall_comparison.png", "Recall (%)")
    plot_single_metric("F1-Score", [80.8, 81.3, 91.1], "f1_comparison.png", "F1-Score (%)")
    plot_multi_metric()
    plot_exam_shift_trajectory()
    print(f"Generated 6 plots in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
