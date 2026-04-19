"""Appendix figures for §4.3: structural graph effects."""

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"
import os

OUTDIR = "figures/appendix"
os.makedirs(OUTDIR, exist_ok=True)

df = pd.read_csv("results/all_results.csv")
synth = df[df["batch"] == "TUNED_BATCH_SYNTH"].copy()

MODELS = ["SimGCL", "MHCN", "LightGCN", "DiffNet", "BPR"]
COLORS = {
    "SimGCL": "#CC79A7",
    "MHCN": "#009E73",
    "LightGCN": "#0072B2",
    "DiffNet": "#D55E00",
    "BPR": "#7F7F7F",
}
MARKERS = {"SimGCL": "o", "MHCN": "D", "LightGCN": "x", "DiffNet": "s", "BPR": "+"}

fig, ax = plt.subplots(figsize=(7, 4))

FAKE_DATASETS = {
    0.05: "fake_users_u2000_i5000_sp0.01_ns0.1_fake0.05",
    0.10: "fake_users_u2000_i5000_sp0.01_ns0.1_fake0.1",
    0.20: "fake_users_u2000_i5000_sp0.01_ns0.1_fake0.2",
    0.40: "fake_users_u2000_i5000_sp0.01_ns0.1_fake0.4",
}
params = sorted(FAKE_DATASETS.keys())

for model in MODELS:
    means = []
    sems = []
    for p in params:
        sub = synth[(synth["dataset"] == FAKE_DATASETS[p]) & (synth["model"] == model)]
        means.append(sub["ndcg@10"].mean())
        sems.append(sub["ndcg@10"].std() / np.sqrt(len(sub)))
    ax.errorbar(
        params,
        means,
        yerr=sems,
        label=model,
        color=COLORS[model],
        marker=MARKERS[model],
        linewidth=1.5,
        markersize=6,
        capsize=3,
    )

ax.set_xlabel(r"Fake-User Fraction $\phi$", fontsize=13)
ax.set_ylabel("NDCG@10", fontsize=13)
ax.set_xticks(params)
ax.set_xticklabels([str(p) for p in params])
ax.tick_params(axis="both", labelsize=11)
ax.legend(fontsize=11, loc="center left", bbox_to_anchor=(1.02, 0.5), framealpha=0.8)
plt.tight_layout()
fig.savefig(f"{OUTDIR}/fake_user_sweep.pdf", bbox_inches="tight", dpi=300)
fig.savefig(f"{OUTDIR}/fake_user_sweep.png", bbox_inches="tight", dpi=150)
plt.close()
print("wrote fake_user_sweep.{pdf,png}")


DEGREE_DATASETS = {
    "Echo Chamber": {
        2.0: "echo_chamber_u2000_i5000_sp0.01_ns0.1_deg2.0",
        4.0: "echo_chamber_u2000_i5000_sp0.01_ns0.1_deg4.0",
        10.0: "echo_chamber_u2000_i5000_sp0.01_ns0.1_deg10.0",
        20.0: "echo_chamber_u2000_i5000_sp0.01_ns0.1_deg20.0",
    },
    "Partial Alignment": {
        2.0: "partial_alignment_u2000_i5000_sp0.01_ns0.1_deg2.0",
        4.0: "partial_alignment_u2000_i5000_sp0.01_ns0.1_deg4.0",
        10.0: "partial_alignment_u2000_i5000_sp0.01_ns0.1_deg10.0",
        20.0: "partial_alignment_u2000_i5000_sp0.01_ns0.1_deg20.0",
    },
    "Contrarian": {
        2.0: "contrarian_u2000_i5000_sp0.01_ns0.1_deg2.0",
        4.0: "contrarian_u2000_i5000_sp0.01_ns0.1_deg4.0",
        10.0: "contrarian_u2000_i5000_sp0.01_ns0.1_deg10.0",
        20.0: "contrarian_u2000_i5000_sp0.01_ns0.1_deg20.0",
    },
    "Random": {
        2.0: "random_u2000_i5000_sp0.01_ns0.1_erp0.001",
        4.0: "random_u2000_i5000_sp0.01_ns0.1_erp0.002",
        10.0: "random_u2000_i5000_sp0.01_ns0.1_erp0.005",
        20.0: "random_u2000_i5000_sp0.01_ns0.1_erp0.01",
    },
}

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
topo_order = ["Echo Chamber", "Partial Alignment", "Contrarian", "Random"]

for ax_idx, topo in enumerate(topo_order):
    ax = axes[ax_idx // 2][ax_idx % 2]
    datasets = DEGREE_DATASETS[topo]
    params = sorted(datasets.keys())

    for model in MODELS:
        means = []
        sems = []
        for p in params:
            sub = synth[(synth["dataset"] == datasets[p]) & (synth["model"] == model)]
            means.append(sub["ndcg@10"].mean())
            sems.append(sub["ndcg@10"].std() / np.sqrt(len(sub)))
        ax.errorbar(
            params,
            means,
            yerr=sems,
            label=model,
            color=COLORS[model],
            marker=MARKERS[model],
            linewidth=1.5,
            markersize=6,
            capsize=3,
        )

    ax.set_xlabel(r"Mean Degree $\bar{k}$", fontsize=13)
    ax.set_ylabel("NDCG@10", fontsize=13)
    ax.set_xticks(params)
    ax.tick_params(axis="both", labelsize=11)
    ax.set_title(topo, fontsize=14)

# shared legend
handles, labels = axes[0][0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.02),
    ncol=5,
    fontsize=12,
    frameon=False,
)
plt.tight_layout(rect=[0, 0.03, 1, 1])
fig.savefig(f"{OUTDIR}/degree_sweep_all_models.pdf", bbox_inches="tight", dpi=300)
fig.savefig(f"{OUTDIR}/degree_sweep_all_models.png", bbox_inches="tight", dpi=150)
plt.close()
print("wrote degree_sweep_all_models.{pdf,png}")
