"""Figures for §4.3: structural graph effects."""

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"
import os

from matplotlib.patches import Patch

OUTDIR = "figures/results"
os.makedirs(OUTDIR, exist_ok=True)

df = pd.read_csv("results/all_results.csv")
synth = df[df["batch"] == "TUNED_BATCH_SYNTH"].copy()
stats_df = pd.read_csv(f"{OUTDIR}/structural_sweep_stats.csv")

MODELS_ORDER = ["SimGCL", "MHCN", "LightGCN", "DiffNet", "BPR"]
SOCIAL_MODELS = {"MHCN", "DiffNet"}

SWEEP_LABELS = {
    "contrarian_degree": "Contrarian\nDegree $\\bar{k}$",
    "echo_chamber_degree": "Echo Chamber\nDegree $\\bar{k}$",
    "partial_alignment_degree": "Partial Align.\nDegree $\\bar{k}$",
    "partial_alignment_sd": "Partial Align.\nShared Dims $d_s$",
    "random_erp": "Random\nDegree $\\bar{k}$",
    "fake_users_fraction": "Fake Users\nFraction $\\phi$",
}

SWEEP_ORDER = [
    "echo_chamber_degree",
    "partial_alignment_degree",
    "contrarian_degree",
    "partial_alignment_sd",
    "random_erp",
    "fake_users_fraction",
]


fig, ax = plt.subplots(figsize=(8, 5))

# significance matrix: 0=ns, 1=raw only, 2=BH-FDR
matrix = np.zeros((len(SWEEP_ORDER), len(MODELS_ORDER)))
for i, sweep in enumerate(SWEEP_ORDER):
    for j, model in enumerate(MODELS_ORDER):
        row = stats_df[(stats_df["sweep"] == sweep) & (stats_df["model"] == model)]
        if len(row) == 0:
            matrix[i, j] = np.nan
        else:
            r = row.iloc[0]
            if r["both_adj"]:
                matrix[i, j] = 2
            elif r["both_raw"]:
                matrix[i, j] = 1
            else:
                matrix[i, j] = 0

from matplotlib.colors import ListedColormap

cmap = ListedColormap(["#f0f0f0", "#ffc107", "#d32f2f"])
im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0, vmax=2)

ax.set_xticks(range(len(MODELS_ORDER)))
ax.set_xticklabels(MODELS_ORDER, fontsize=10, rotation=45, ha="right")
ax.set_yticks(range(len(SWEEP_ORDER)))
ax.set_yticklabels([SWEEP_LABELS[s] for s in SWEEP_ORDER], fontsize=9)

for i in range(len(SWEEP_ORDER)):
    for j in range(len(MODELS_ORDER)):
        row = stats_df[
            (stats_df["sweep"] == SWEEP_ORDER[i]) & (stats_df["model"] == MODELS_ORDER[j])
        ]
        if len(row) > 0:
            r = row.iloc[0]
            if r["both_adj"]:
                direction = "+" if r["boot_diff"] > 0 else "-" if r["boot_diff"] < 0 else ""
                ax.text(
                    j,
                    i,
                    direction,
                    ha="center",
                    va="center",
                    fontsize=14,
                    fontweight="bold",
                    color="white",
                )

# mark social models with asterisk in x-tick labels
xlabels = [f"{m}*" if m in SOCIAL_MODELS else m for m in MODELS_ORDER]
ax.set_xticklabels(xlabels, fontsize=10, rotation=45, ha="right")

# horizontal lines between groups
ax.axhline(y=3.5, color="black", linewidth=1.5, linestyle="--")
ax.axhline(y=4.5, color="black", linewidth=1.5, linestyle="--")

# group labels on right
ax.text(
    len(MODELS_ORDER) + 0.3,
    1.5,
    "Group II\n(coupled)",
    fontsize=8,
    va="center",
    ha="left",
    color="#666",
    family="serif",
)
ax.text(
    len(MODELS_ORDER) + 0.3,
    4.0,
    "Group I",
    fontsize=8,
    va="center",
    ha="left",
    color="#666",
    family="serif",
)
ax.text(
    len(MODELS_ORDER) + 0.3,
    5.0,
    "Group III",
    fontsize=8,
    va="center",
    ha="left",
    color="#666",
    family="serif",
)

legend_elements = [
    Patch(facecolor="#f0f0f0", edgecolor="gray", label="Not significant"),
    Patch(facecolor="#d32f2f", edgecolor="gray", label="BH-FDR adjusted $p < 0.05$"),
    Patch(facecolor="none", edgecolor="none", label="* = social-aware model"),
]
ax.legend(
    handles=legend_elements,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.25),
    ncol=3,
    fontsize=9,
    frameon=False,
)

ax.set_title("Structural Sweep Significance (LMM + Spearman)", fontsize=12, pad=15)
plt.tight_layout(rect=[0, 0.08, 0.92, 1])
fig.savefig(f"{OUTDIR}/structural_sweep_heatmap.pdf", bbox_inches="tight", dpi=300)
fig.savefig(f"{OUTDIR}/structural_sweep_heatmap.png", bbox_inches="tight", dpi=150)
plt.close()
print("wrote structural_sweep_heatmap.{pdf,png}")


SWEEP_DATASETS = {
    "contrarian_degree": {
        2.0: "contrarian_u2000_i5000_sp0.01_ns0.1_deg2.0",
        4.0: "contrarian_u2000_i5000_sp0.01_ns0.1_deg4.0",
        10.0: "contrarian_u2000_i5000_sp0.01_ns0.1_deg10.0",
        20.0: "contrarian_u2000_i5000_sp0.01_ns0.1_deg20.0",
    },
    "echo_chamber_degree": {
        2.0: "echo_chamber_u2000_i5000_sp0.01_ns0.1_deg2.0",
        4.0: "echo_chamber_u2000_i5000_sp0.01_ns0.1_deg4.0",
        10.0: "echo_chamber_u2000_i5000_sp0.01_ns0.1_deg10.0",
        20.0: "echo_chamber_u2000_i5000_sp0.01_ns0.1_deg20.0",
    },
    "partial_alignment_degree": {
        2.0: "partial_alignment_u2000_i5000_sp0.01_ns0.1_deg2.0",
        4.0: "partial_alignment_u2000_i5000_sp0.01_ns0.1_deg4.0",
        10.0: "partial_alignment_u2000_i5000_sp0.01_ns0.1_deg10.0",
        20.0: "partial_alignment_u2000_i5000_sp0.01_ns0.1_deg20.0",
    },
    "partial_alignment_sd": {
        1: "partial_alignment_u2000_i5000_sp0.01_ns0.1_sd1",
        3: "partial_alignment_u2000_i5000_sp0.01_ns0.1_sd3",
        7: "partial_alignment_u2000_i5000_sp0.01_ns0.1_sd7",
        9: "partial_alignment_u2000_i5000_sp0.01_ns0.1_sd9",
    },
    "random_erp": {
        2.0: "random_u2000_i5000_sp0.01_ns0.1_erp0.001",
        4.0: "random_u2000_i5000_sp0.01_ns0.1_erp0.002",
        10.0: "random_u2000_i5000_sp0.01_ns0.1_erp0.005",
        20.0: "random_u2000_i5000_sp0.01_ns0.1_erp0.01",
    },
}

# SSL-tier models show no effect on the degree sweeps, so only GCN+MF tiers are plotted
TIER_COLORS = {
    "SimGCL": "#CC79A7",
    "MHCN": "#009E73",
    "DiffNet": "#D55E00",
    "LightGCN": "#0072B2",
    "BPR": "#7F7F7F",
}
TIER_MARKERS = {
    "SimGCL": "v",
    "MHCN": "D",
    "DiffNet": "s",
    "LightGCN": "x",
    "BPR": "+",
}

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)
degree_sweeps = ["echo_chamber_degree", "partial_alignment_degree", "contrarian_degree"]
degree_titles = ["Echo Chamber", "Partial Alignment", "Contrarian"]
plot_models = ["DiffNet", "LightGCN", "BPR"]

for ax_idx, (sweep, title) in enumerate(zip(degree_sweeps, degree_titles)):
    ax = axes[ax_idx]
    datasets = SWEEP_DATASETS[sweep]
    params = sorted(datasets.keys())

    for model in plot_models:
        means = []
        sems = []
        for p in params:
            sub = synth[(synth["dataset"] == datasets[p]) & (synth["model"] == model)]
            means.append(sub["ndcg@10"].mean())
            sems.append(sub["ndcg@10"].std() / np.sqrt(len(sub)))

        is_sig = stats_df[(stats_df["sweep"] == sweep) & (stats_df["model"] == model)]
        lw = 2
        alpha = 1.0

        ax.errorbar(
            params,
            means,
            yerr=sems,
            label=model,
            color=TIER_COLORS[model],
            marker=TIER_MARKERS[model],
            linewidth=lw,
            alpha=alpha,
            markersize=6,
            capsize=3,
        )

    ax.set_xlabel(r"Mean Degree $\bar{k}$", fontsize=13)
    ax.set_title(title, fontsize=14)
    if ax_idx == 0:
        ax.set_ylabel("NDCG@10", fontsize=13)
    ax.set_xticks(params)
    ax.tick_params(axis="both", labelsize=11)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.08),
    ncol=len(plot_models),
    fontsize=12,
    frameon=False,
)
plt.tight_layout()
fig.savefig(f"{OUTDIR}/degree_sweep_profiles.pdf", bbox_inches="tight", dpi=300)
fig.savefig(f"{OUTDIR}/degree_sweep_profiles.png", bbox_inches="tight", dpi=150)
plt.close()
print("wrote degree_sweep_profiles.{pdf,png}")


# MF-tier models only; LightGCN is omitted so the DiffNet effect is not compressed
other_plot_models = ["DiffNet", "BPR"]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# ERP sweep
ax = axes[0]
datasets = SWEEP_DATASETS["random_erp"]
params = sorted(datasets.keys())
for model in other_plot_models:
    means = []
    sems = []
    for p in params:
        sub = synth[(synth["dataset"] == datasets[p]) & (synth["model"] == model)]
        means.append(sub["ndcg@10"].mean())
        sems.append(sub["ndcg@10"].std() / np.sqrt(len(sub)))
    is_sig = stats_df[(stats_df["sweep"] == "random_erp") & (stats_df["model"] == model)]
    lw = 2
    alpha = 1.0
    ax.errorbar(
        params,
        means,
        yerr=sems,
        label=model,
        color=TIER_COLORS[model],
        marker=TIER_MARKERS[model],
        linewidth=lw,
        alpha=alpha,
        markersize=6,
        capsize=3,
    )
ax.set_xlabel(r"Mean Degree $\bar{k}$", fontsize=13)
ax.set_ylabel("NDCG@10", fontsize=13)
ax.set_title(r"Random ($\bar{k}$ Sweep)", fontsize=14)
ax.tick_params(axis="both", labelsize=11)
ax.legend(fontsize=12, loc="best", framealpha=0.8)

# SD sweep
ax = axes[1]
datasets = SWEEP_DATASETS["partial_alignment_sd"]
params = sorted(datasets.keys())
for model in other_plot_models:
    means = []
    sems = []
    for p in params:
        sub = synth[(synth["dataset"] == datasets[p]) & (synth["model"] == model)]
        means.append(sub["ndcg@10"].mean())
        sems.append(sub["ndcg@10"].std() / np.sqrt(len(sub)))
    is_sig = stats_df[(stats_df["sweep"] == "partial_alignment_sd") & (stats_df["model"] == model)]
    lw = 2
    alpha = 1.0
    ax.errorbar(
        params,
        means,
        yerr=sems,
        label=model,
        color=TIER_COLORS[model],
        marker=TIER_MARKERS[model],
        linewidth=lw,
        alpha=alpha,
        markersize=6,
        capsize=3,
    )
ax.set_xlabel(r"Shared Dimensions $d_s$", fontsize=13)
ax.set_title(r"Partial Alignment ($d_s$ Sweep)", fontsize=14)
ax.tick_params(axis="both", labelsize=11)
ax.legend(fontsize=12, loc="best", framealpha=0.8)

plt.tight_layout()
fig.savefig(f"{OUTDIR}/other_sweep_profiles.pdf", bbox_inches="tight", dpi=300)
fig.savefig(f"{OUTDIR}/other_sweep_profiles.png", bbox_inches="tight", dpi=150)
plt.close()
print("wrote other_sweep_profiles.{pdf,png}")


print("\nLATEX TABLE: BH-FDR Significant Cells")
print(r"""
\begin{table}[t]
\centering
\caption{Structural sweep cells that remain significant after BH-FDR adjustment on both the mixed-effects slope and Spearman correlation ($q=0.05$). The ``Group'' column indicates whether the sweep is subject to the interaction-graph coupling described in Section \ref{sec:interaction_coupling}. All seven significant cells correspond to social-aware models; no non-social baseline achieves significance on any structural sweep. The bootstrap 95\% CI is computed on the endpoint difference (highest $-$ lowest sweep value) with pairing preserved across seeds.}
\label{tab:structural_sig}
\resizebox{\textwidth}{!}{%
\begin{tabular}{llllrrrrr}
\toprule
\textbf{Sweep} & \textbf{Group} & \textbf{Model} & \textbf{LMM slope} & \textbf{LMM $p_{\mathrm{adj}}$} & \textbf{Spearman $\rho$} & \textbf{Spearman $p_{\mathrm{adj}}$} & \textbf{$\Delta$NDCG} & \textbf{95\% CI} \\
\midrule""")

sig_rows = stats_df[stats_df["both_adj"]].sort_values(["sweep", "model"])
for _, r in sig_rows.iterrows():
    sweep_nice = (
        r["sweep"]
        .replace("_", " ")
        .replace("degree", "deg.")
        .replace("partial alignment", "partial align.")
        .title()
    )
    group = r["group"]
    ci = f"[{r['boot_ci_lo']:.4f}, {r['boot_ci_hi']:.4f}]"
    print(
        f"{sweep_nice} & {group} & {r['model']} & {r['lmm_slope']:.6f} & {r['lmm_p_adj']:.4f} & {r['spearman_rho']:.3f} & {r['spearman_p_adj']:.4f} & {r['boot_diff']:.4f} & {ci} \\\\"
    )

print(r"""\bottomrule
\end{tabular}%
}
\end{table}""")

print("\nDone.")
