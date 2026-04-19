"""Figure for §4.4: NDCG@10 vs each shared generation parameter."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BASE = Path("/n/fs/recbench/new_rec/results/TUNED_BATCH_SYNTH")
OUT_DIR = Path("/n/fs/recbench/new_rec/figures/results")

MODELS = ["BPR", "DiffNet", "LightGCN", "MHCN", "SimGCL"]
DSEEDS = ["dseed_1", "dseed_2", "dseed_3"]
MSEEDS = ["mseed_1", "mseed_12", "mseed_123"]

TOPOLOGIES = [
    "scale_free",
    "small_world",
    "random",
    "echo_chamber",
    "partial_alignment",
    "contrarian",
    "star",
    "fake_users",
]

MODEL_COLORS = {
    "BPR": "#7F7F7F",
    "DiffNet": "#D55E00",
    "LightGCN": "#0072B2",
    "MHCN": "#009E73",
    "SimGCL": "#CC79A7",
}

RHO_SWEEP = [0.003, 0.005, 0.01, 0.02, 0.05]
ETA_SWEEP = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
M_SWEEP = [500, 2000, 5000]
N_SWEEP = [2000, 5000, 10000]


def format_num(x: float) -> str:
    s = f"{x:g}"
    return s


def config_dir(topo: str, M: int, N: int, rho: float, eta: float) -> str:
    suffix = ""
    if topo == "fake_users":
        suffix = "_fake0.1"
    return f"{topo}_u{M}_i{N}_sp{format_num(rho)}_ns{format_num(eta)}{suffix}"


def load_cell(topo: str, M: int, N: int, rho: float, eta: float, model: str) -> list[float]:
    vals = []
    for dseed in DSEEDS:
        for mseed in MSEEDS:
            path = (
                BASE
                / dseed
                / mseed
                / config_dir(topo, M, N, rho, eta)
                / model
                / "final_metrics.csv"
            )
            if not path.exists():
                continue
            df = pd.read_csv(path)
            if df.empty or "ndcg@10" not in df.columns:
                continue
            vals.append(float(df["ndcg@10"].iloc[0]))
    return vals


def sweep_means(
    sweep_name: str,
    level_list,
    fixed_M: int,
    fixed_N: int,
    fixed_rho: float,
    fixed_eta: float,
) -> pd.DataFrame:
    """Build a (level, model) mean NDCG table averaged across 8 topologies and seeds."""
    rows = []
    for level in level_list:
        M, N, rho, eta = fixed_M, fixed_N, fixed_rho, fixed_eta
        if sweep_name == "rho":
            rho = level
        elif sweep_name == "eta":
            eta = level
        elif sweep_name == "M":
            M = level
        elif sweep_name == "N":
            N = level
        else:
            raise ValueError(sweep_name)
        for model in MODELS:
            all_vals = []
            for topo in TOPOLOGIES:
                all_vals.extend(load_cell(topo, M, N, rho, eta, model))
            if all_vals:
                rows.append(
                    {
                        "sweep": sweep_name,
                        "level": level,
                        "model": model,
                        "mean_ndcg": sum(all_vals) / len(all_vals),
                        "n": len(all_vals),
                    }
                )
    return pd.DataFrame(rows)


def plot(all_df: dict[str, pd.DataFrame]) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "font.size": 13,
            "axes.titlesize": 14,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    panels = [
        ("rho", r"Sparsity $\rho$", axes[0, 0], True),
        ("eta", r"Noise fraction $\eta$", axes[0, 1], False),
        ("M", r"Users $M$", axes[1, 0], False),
        ("N", r"Items $N$", axes[1, 1], False),
    ]

    for sweep_name, xlabel, ax, log_x in panels:
        df = all_df[sweep_name]
        for model in MODELS:
            sub = df[df["model"] == model].sort_values("level")
            if sub.empty:
                continue
            ax.plot(
                sub["level"],
                sub["mean_ndcg"],
                "-o",
                color=MODEL_COLORS[model],
                linewidth=2,
                markersize=6,
                label=model,
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("NDCG@10")
        if log_x:
            ax.set_xscale("log")
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
    )

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"shared_sweep_overview.{ext}", bbox_inches="tight", dpi=200)
    print(f"wrote {OUT_DIR}/shared_sweep_overview.{{pdf,png}}")


def main() -> None:
    all_df = {}
    all_df["rho"] = sweep_means("rho", RHO_SWEEP, 2000, 5000, 0.01, 0.1)
    all_df["eta"] = sweep_means("eta", ETA_SWEEP, 2000, 5000, 0.01, 0.1)
    all_df["M"] = sweep_means("M", M_SWEEP, 2000, 5000, 0.01, 0.1)
    all_df["N"] = sweep_means("N", N_SWEEP, 2000, 5000, 0.01, 0.1)
    for sweep, df in all_df.items():
        print(f"\n== {sweep} ==")
        print(df.pivot(index="level", columns="model", values="mean_ndcg").round(4))
    plot(all_df)


if __name__ == "__main__":
    main()
