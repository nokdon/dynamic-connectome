"""Export static figures directly from the published evaluation report."""

import json
from pathlib import Path

import numpy as np

from .artifacts import load_model, load_problem


def plot_report(report_path, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    report = json.loads(Path(report_path).read_text())
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    names = ["dense_soft", "dense", "learned", "heuristic", "random"]
    labels = [
        "Dense soft · 132 edges",
        "Fixed dense · 132 edges",
        "Learned sparse · 44 edges",
        "Shortest-path · 44 edges",
        "Random sparse · 44 edges",
    ]
    if any(n not in report["summary"] for n in names):
        raise ValueError("The comparison plot requires all five structural families")
    fig, ax = plt.subplots(figsize=(9, 4.6), layout="constrained")
    fig.patch.set_facecolor("#fafbfd")
    ax.set_facecolor("#fafbfd")
    means = [report["summary"][n]["mean_mse"] for n in names]
    std = [report["summary"][n]["std_across_seeds"] for n in names]
    colors = ["#90a7bc", "#90a7bc", "#137c87", "#bbc5ce", "#bbc5ce"]
    ax.barh(labels, means, xerr=std, color=colors, height=0.6, capsize=4)
    ax.invert_yaxis()
    ax.set_xlim(0, max(m + s for m, s in zip(means, std)) * 1.22)
    for i, (m, s) in enumerate(zip(means, std)):
        ax.text(m + s + 0.014, i, f"{m:.3f}", va="center", fontsize=11)
    ax.set_xlabel("Mean squared error ↓")
    ax.set_title(
        "One third of the edges; a measurable routing trade-off",
        loc="left",
        fontweight="bold",
        pad=18,
    )
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", alpha=0.13)
    ax.set_axisbelow(True)
    fig.suptitle(
        f"{report['batch_size'] * report['batches']:,} common synthetic examples · historical checkpoints · bars show seed SD",
        fontsize=9,
        y=1.03,
        color="#52616b",
    )
    for ext in ("svg", "png"):
        path = output / f"checkpoint_comparison.{ext}"
        if path.exists():
            raise FileExistsError(path)
        fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    problem = load_problem()
    q = np.asarray(problem["q"])
    _, p, _, _ = load_model("learned-0")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), layout="constrained")
    for ax, z, title, color in zip(
        axes,
        [np.asarray(problem["M"]) > 0, np.asarray(p) > 0],
        ["Candidate graph · 132 directed edges", "Selected seed 0 · 44 directed edges"],
        ["#99aabb", "#137c87"],
    ):
        for s, t in zip(*np.where(z)):
            ax.add_patch(
                FancyArrowPatch(
                    q[s],
                    q[t],
                    arrowstyle="-|>",
                    mutation_scale=7,
                    linewidth=0.8,
                    color=color,
                    alpha=0.6,
                    shrinkA=5,
                    shrinkB=5,
                    connectionstyle="arc3,rad=0.08",
                )
            )
        ax.scatter(q[:, 0], q[:, 1], s=85, c="#173447", edgecolors="white", zorder=3)
        for i, xy in enumerate(q):
            ax.annotate(
                str(i), xy, xytext=(5, 5), textcoords="offset points", fontsize=7
            )
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_aspect("equal")
        ax.set_xlim(q[:, 0].min() - 0.08, q[:, 0].max() + 0.08)
        ax.set_ylim(q[:, 1].min() - 0.08, q[:, 1].max() + 0.08)
        ax.axis("off")
    for ext in ("svg", "png"):
        path = output / f"routing_graphs.{ext}"
        if path.exists():
            raise FileExistsError(path)
        fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
