"""Figure for §4.4: line-graph result overlaid on sparsity sweep."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BASE = Path("/n/fs/recbench/new_rec/results/TUNED_BATCH_SYNTH")
DATASETS = Path("/n/fs/recbench/new_rec/datasets")
OUT_DIR = Path("/n/fs/recbench/new_rec/figures/results")

MODELS = ["SimGCL", "LightGCN"]
DSEEDS = ["dseed_1", "dseed_2", "dseed_3"]
MSEEDS = ["mseed_1", "mseed_12", "mseed_123"]

SWEEP_TOPOLOGIES = [
    "scale_free",
    "small_world",
    "random",
    "echo_chamber",
    "partial_alignment",
    "contrarian",
    "star",
    "fake_users",
]

RHO_SWEEP = [0.003, 0.005, 0.01, 0.02, 0.05]
N_FIXED = 5000
M_FIXED = 2000

MODEL_COLORS = {
    "SimGCL": "#CC79A7",
    "LightGCN": "#0072B2",
}


def fmt(x: float) -> str:
    return f"{x:g}"


def sweep_config_dir(topo: str, rho: float) -> str:
    suffix = "_fake0.1" if topo == "fake_users" else ""
    return f"{topo}_u{M_FIXED}_i{N_FIXED}_sp{fmt(rho)}_ns0.1{suffix}"


def load_ndcg(dseed: str, mseed: str, config: str, model: str) -> float | None:
    path = BASE / dseed / mseed / config / model / "final_metrics.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty or "ndcg@10" not in df.columns:
        return None
    return float(df["ndcg@10"].iloc[0])


def sweep_point(rho: float, model: str) -> float | None:
    vals = []
    for dseed in DSEEDS:
        for mseed in MSEEDS:
            for topo in SWEEP_TOPOLOGIES:
                v = load_ndcg(dseed, mseed, sweep_config_dir(topo, rho), model)
                if v is not None:
                    vals.append(v)
    return sum(vals) / len(vals) if vals else None


def line_graph_point(model: str) -> tuple[float, float] | None:
    config = "line_graph_u2000_i5000_sp0.01_ns0.1_max30_decay0.5"
    ndcg_vals = []
    for dseed in DSEEDS:
        for mseed in MSEEDS:
            v = load_ndcg(dseed, mseed, config, model)
            if v is not None:
                ndcg_vals.append(v)
    if not ndcg_vals:
        return None

    counts = []
    for dseed in DSEEDS:
        inter_path = DATASETS / dseed / config / f"{config}.inter"
        if inter_path.exists():
            with open(inter_path) as f:
                n_lines = sum(1 for _ in f) - 1
            counts.append(n_lines / M_FIXED)
    mean_inter = sum(counts) / len(counts) if counts else 20.0
    return mean_inter, sum(ndcg_vals) / len(ndcg_vals)


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "font.size": 15,
            "axes.titlesize": 16,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

    for ax, model in zip(axes, MODELS):
        color = MODEL_COLORS[model]
        xs = [N_FIXED * rho for rho in RHO_SWEEP]
        ys = [sweep_point(rho, model) for rho in RHO_SWEEP]
        ax.plot(
            xs,
            ys,
            "-o",
            color=color,
            linewidth=2,
            markersize=6,
            label="Sparsity sweep",
        )

        pt = line_graph_point(model)
        if pt is not None:
            x_lg, y_lg = pt
            ax.plot(
                x_lg,
                y_lg,
                marker="*",
                markersize=24,
                markerfacecolor=color,
                markeredgecolor="black",
                markeredgewidth=1.2,
                linestyle="none",
                label="Line graph",
            )
            print(f"{model}: line graph at ({x_lg:.1f}, {y_lg:.3f})")

        ax.set_xscale("log")
        sweep_ticks = [N_FIXED * rho for rho in RHO_SWEEP]
        ax.set_xticks(sweep_ticks)
        ax.set_xticklabels([str(int(t)) for t in sweep_ticks])
        ax.minorticks_off()
        ax.set_xlabel(r"Mean interactions per user ($N\rho$)")
        ax.set_ylabel("NDCG@10")
        ax.set_title(model, color="black")
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="best", frameon=False)

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"line_graph_density.{ext}", bbox_inches="tight", dpi=200)
    print(f"wrote {OUT_DIR}/line_graph_density.{{pdf,png}}")


if __name__ == "__main__":
    main()
