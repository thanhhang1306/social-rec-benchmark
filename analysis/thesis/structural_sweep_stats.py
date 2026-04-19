"""Statistics for §4.3: LMM, Spearman, and paired-bootstrap tests on the structural sweep (BH-FDR corrected)."""

import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")
np.random.seed(42)

OUTDIR = "figures/results"
os.makedirs(OUTDIR, exist_ok=True)

df = pd.read_csv("results/all_results.csv")
synth = df[df["batch"] == "TUNED_BATCH_SYNTH"].copy()

MODELS = ["SimGCL", "MHCN", "LightGCN", "ASLGCN", "CSGCN", "HAGN", "DiffNet", "BPR"]
SOCIAL_MODELS = ["MHCN", "ASLGCN", "CSGCN", "HAGN", "DiffNet"]
NONSOCIAL_MODELS = ["SimGCL", "LightGCN", "BPR"]

SWEEPS = {}

# degree sweeps (Group II)
for topo in ["echo_chamber", "partial_alignment", "contrarian"]:
    sweep_name = f"{topo}_degree"
    datasets = {
        2.0: f"{topo}_u2000_i5000_sp0.01_ns0.1_deg2.0",
        4.0: f"{topo}_u2000_i5000_sp0.01_ns0.1_deg4.0",
        10.0: f"{topo}_u2000_i5000_sp0.01_ns0.1_deg10.0",
        20.0: f"{topo}_u2000_i5000_sp0.01_ns0.1_deg20.0",
    }
    SWEEPS[sweep_name] = {"param": "degree", "datasets": datasets, "group": "II"}

# erp sweep (Group I: random)
SWEEPS["random_erp"] = {
    "param": "erp",
    "datasets": {
        0.001: "random_u2000_i5000_sp0.01_ns0.1_erp0.001",
        0.002: "random_u2000_i5000_sp0.01_ns0.1_erp0.002",
        0.005: "random_u2000_i5000_sp0.01_ns0.1_erp0.005",
        0.01: "random_u2000_i5000_sp0.01_ns0.1_erp0.01",
    },
    "group": "I",
}

# sd sweep (Group II: partial_alignment)
SWEEPS["partial_alignment_sd"] = {
    "param": "sd",
    "datasets": {
        1: "partial_alignment_u2000_i5000_sp0.01_ns0.1_sd1",
        3: "partial_alignment_u2000_i5000_sp0.01_ns0.1_sd3",
        7: "partial_alignment_u2000_i5000_sp0.01_ns0.1_sd7",
        9: "partial_alignment_u2000_i5000_sp0.01_ns0.1_sd9",
    },
    "group": "II",
}

# fake user sweep (Group III)
SWEEPS["fake_users_fraction"] = {
    "param": "fake_fraction",
    "datasets": {
        0.05: "fake_users_u2000_i5000_sp0.01_ns0.1_fake0.05",
        0.1: "fake_users_u2000_i5000_sp0.01_ns0.1_fake0.1",
        0.2: "fake_users_u2000_i5000_sp0.01_ns0.1_fake0.2",
        0.4: "fake_users_u2000_i5000_sp0.01_ns0.1_fake0.4",
    },
    "group": "III",
}


def build_sweep_df(sweep_config, model):
    """Build a DataFrame with sweep_param, ndcg@10, dseed, mseed for one model."""
    rows = []
    for param_val, dataset_name in sweep_config["datasets"].items():
        sub = synth[(synth["dataset"] == dataset_name) & (synth["model"] == model)]
        for _, row in sub.iterrows():
            rows.append(
                {
                    "sweep_param": float(param_val),
                    "ndcg10": row["ndcg@10"],
                    "dseed": str(int(row["dseed"])),
                    "mseed": str(int(row["mseed"])),
                }
            )
    return pd.DataFrame(rows)


def run_lmm(sdf):
    """Run linear mixed-effects model. Returns slope, p-value."""
    if len(sdf) < 8:
        return np.nan, np.nan, np.nan
    try:
        model = MixedLM.from_formula(
            "ndcg10 ~ sweep_param",
            groups="dseed",
            re_formula="1",
            vc_formula={"mseed": "0 + C(mseed)"},
            data=sdf,
        )
        result = model.fit(reml=True)
        slope = result.fe_params["sweep_param"]
        pval = result.pvalues["sweep_param"]
        se = result.bse["sweep_param"]
        return slope, pval, se
    except Exception:
        try:
            model = MixedLM.from_formula(
                "ndcg10 ~ sweep_param", groups="dseed", re_formula="1", data=sdf
            )
            result = model.fit(reml=True)
            slope = result.fe_params["sweep_param"]
            pval = result.pvalues["sweep_param"]
            se = result.bse["sweep_param"]
            return slope, pval, se
        except Exception:
            return np.nan, np.nan, np.nan


def run_spearman(sdf):
    """Run Spearman rank correlation. Returns rho, p-value."""
    if len(sdf) < 4:
        return np.nan, np.nan
    rho, pval = stats.spearmanr(sdf["sweep_param"], sdf["ndcg10"])
    return rho, pval


def run_bootstrap(sdf, n_boot=5000):
    """Paired bootstrap on endpoint differences. Returns mean_diff, CI_lo, CI_hi."""
    params = sorted(sdf["sweep_param"].unique())
    if len(params) < 2:
        return np.nan, np.nan, np.nan
    lo_param, hi_param = params[0], params[-1]
    lo_df = sdf[sdf["sweep_param"] == lo_param].sort_values(["dseed", "mseed"])
    hi_df = sdf[sdf["sweep_param"] == hi_param].sort_values(["dseed", "mseed"])

    # merge on dseed+mseed for pairing
    merged = lo_df.merge(hi_df, on=["dseed", "mseed"], suffixes=("_lo", "_hi"))
    if len(merged) == 0:
        return np.nan, np.nan, np.nan

    diffs = merged["ndcg10_hi"].values - merged["ndcg10_lo"].values
    mean_diff = np.mean(diffs)

    boot_means = []
    n = len(diffs)
    for _ in range(n_boot):
        idx = np.random.randint(0, n, size=n)
        boot_means.append(np.mean(diffs[idx]))

    ci_lo = np.percentile(boot_means, 2.5)
    ci_hi = np.percentile(boot_means, 97.5)
    return mean_diff, ci_lo, ci_hi


results = []

for sweep_name, sweep_config in SWEEPS.items():
    for model in MODELS:
        sdf = build_sweep_df(sweep_config, model)
        if len(sdf) == 0:
            continue

        lmm_slope, lmm_p, lmm_se = run_lmm(sdf)
        spearman_rho, spearman_p = run_spearman(sdf)
        boot_diff, boot_lo, boot_hi = run_bootstrap(sdf)

        results.append(
            {
                "sweep": sweep_name,
                "group": sweep_config["group"],
                "param": sweep_config["param"],
                "model": model,
                "social": model in SOCIAL_MODELS,
                "n_obs": len(sdf),
                "lmm_slope": lmm_slope,
                "lmm_p": lmm_p,
                "lmm_se": lmm_se,
                "spearman_rho": spearman_rho,
                "spearman_p": spearman_p,
                "boot_diff": boot_diff,
                "boot_ci_lo": boot_lo,
                "boot_ci_hi": boot_hi,
            }
        )

res_df = pd.DataFrame(results)

lmm_mask = res_df["lmm_p"].notna()
spearman_mask = res_df["spearman_p"].notna()

lmm_pvals = res_df.loc[lmm_mask, "lmm_p"].values
_, lmm_adj, _, _ = multipletests(lmm_pvals, alpha=0.05, method="fdr_bh")
res_df.loc[lmm_mask, "lmm_p_adj"] = lmm_adj

sp_pvals = res_df.loc[spearman_mask, "spearman_p"].values
_, sp_adj, _, _ = multipletests(sp_pvals, alpha=0.05, method="fdr_bh")
res_df.loc[spearman_mask, "spearman_p_adj"] = sp_adj

res_df["lmm_sig_raw"] = res_df["lmm_p"] < 0.05
res_df["spearman_sig_raw"] = res_df["spearman_p"] < 0.05
res_df["lmm_sig_adj"] = res_df["lmm_p_adj"] < 0.05
res_df["spearman_sig_adj"] = res_df["spearman_p_adj"] < 0.05
res_df["both_raw"] = res_df["lmm_sig_raw"] & res_df["spearman_sig_raw"]
res_df["both_adj"] = res_df["lmm_sig_adj"] & res_df["spearman_sig_adj"]

print(f"Total tests: {len(res_df)}")
print(f"Tests with both raw p < 0.05: {res_df['both_raw'].sum()}")
print(f"Tests with both adj p < 0.05 (BH-FDR): {res_df['both_adj'].sum()}")

print("\nCELLS WITH BOTH RAW p < 0.05")
raw_sig = res_df[res_df["both_raw"]].sort_values(["sweep", "model"])
for _, r in raw_sig.iterrows():
    adj_note = " ** SIGNIFICANT AFTER BH-FDR **" if r["both_adj"] else " (fails BH-FDR)"
    print(
        f"  {r['sweep']:30s} {r['model']:10s} social={r['social']}  "
        f"LMM slope={r['lmm_slope']:.6f} p={r['lmm_p']:.4f} adj={r['lmm_p_adj']:.4f}  "
        f"Spearman rho={r['spearman_rho']:.4f} p={r['spearman_p']:.4f} adj={r['spearman_p_adj']:.4f}  "
        f"Boot diff={r['boot_diff']:.6f} [{r['boot_ci_lo']:.6f}, {r['boot_ci_hi']:.6f}]"
        f"{adj_note}"
    )

print("\nCELLS SIGNIFICANT AFTER BH-FDR")
adj_sig = res_df[res_df["both_adj"]].sort_values(["sweep", "model"])
if len(adj_sig) == 0:
    print("  NONE")
else:
    for _, r in adj_sig.iterrows():
        print(
            f"  {r['sweep']:30s} {r['model']:10s} social={r['social']}  "
            f"LMM slope={r['lmm_slope']:.6f} adj_p={r['lmm_p_adj']:.4f}  "
            f"Spearman rho={r['spearman_rho']:.4f} adj_p={r['spearman_p_adj']:.4f}  "
            f"Boot diff={r['boot_diff']:.6f} [{r['boot_ci_lo']:.6f}, {r['boot_ci_hi']:.6f}]"
        )

print("\nSOCIAL-SPECIFIC ANALYSIS")
for _, r in adj_sig.iterrows():
    if r["social"]:
        same_sweep_nonsocial = res_df[(res_df["sweep"] == r["sweep"]) & (~res_df["social"])]
        print(f"\n  {r['model']} on {r['sweep']}:")
        for _, ns in same_sweep_nonsocial.iterrows():
            sig_str = "BH-FDR" if ns["both_adj"] else ("raw" if ns["both_raw"] else "ns")
            print(
                f"    {ns['model']:10s} LMM slope={ns['lmm_slope']:.6f} p_adj={ns['lmm_p_adj']:.4f}  "
                f"Spearman rho={ns['spearman_rho']:.4f} p_adj={ns['spearman_p_adj']:.4f}  "
                f"Boot diff={ns['boot_diff']:.6f} [{ns['boot_ci_lo']:.6f}, {ns['boot_ci_hi']:.6f}]  "
                f"sig={sig_str}"
            )

res_df.to_csv(f"{OUTDIR}/structural_sweep_stats.csv", index=False)

print("\n\nFULL TABLE (all sweeps × models)")
print(
    f"{'Sweep':<30s} {'Model':<10s} {'Social':>6s} {'LMM slope':>12s} {'LMM p_adj':>10s} {'Sp rho':>8s} {'Sp p_adj':>10s} {'Boot diff':>12s} {'95% CI':>24s}"
)
print("-" * 130)
for _, r in res_df.sort_values(["sweep", "model"]).iterrows():
    ci_str = (
        f"[{r['boot_ci_lo']:.5f}, {r['boot_ci_hi']:.5f}]"
        if not np.isnan(r["boot_ci_lo"])
        else "---"
    )
    marker = ""
    if r["both_adj"]:
        marker = " **"
    elif r["both_raw"]:
        marker = " *"
    print(
        f"{r['sweep']:<30s} {r['model']:<10s} {str(r['social']):>6s} {r['lmm_slope']:>12.6f} {r['lmm_p_adj']:>10.4f} {r['spearman_rho']:>8.4f} {r['spearman_p_adj']:>10.4f} {r['boot_diff']:>12.6f} {ci_str:>24s}{marker}"
    )
