"""Tables for §4.7: mode hyperparameter configurations across mseeds."""

import os
from collections import Counter

import yaml

MSEEDS = ["mseed_1", "mseed_12", "mseed_123"]
ESTABLISHED = ["BPR", "LightGCN", "SimGCL", "DiffNet", "MHCN"]
HYBRIDS = ["HAGN", "CSGCN", "ASLGCN"]
MODELS = ESTABLISHED + HYBRIDS

SYNTH_BASE = "/n/fs/recbench/new_rec/results/TUNED_BATCH_SYNTH/dseed_1"
REAL_BASE = "/n/fs/recbench/new_rec/results/TUNED_BATCH_REAL"
REAL_DATASETS = ["yelp", "douban-book", "lastfm"]

PRIMARY_TOPOS = [
    ("Scale-Free", "scale_free_u2000_i5000_sp0.01_ns0.1"),
    ("Small-World", "small_world_u2000_i5000_sp0.01_ns0.1"),
    ("Random", "random_u2000_i5000_sp0.01_ns0.1"),
    ("Echo Chamber", "echo_chamber_u2000_i5000_sp0.01_ns0.1"),
    ("Partial Alignment", "partial_alignment_u2000_i5000_sp0.01_ns0.1"),
    ("Contrarian", "contrarian_u2000_i5000_sp0.01_ns0.1"),
    ("Star", "star_u2000_i5000_sp0.01_ns0.1"),
    ("Line", "line_graph_u2000_i5000_sp0.01_ns0.1_max30_decay0.5"),
    ("Fake Users", "fake_users_u2000_i5000_sp0.01_ns0.1_fake0.1"),
]


def load_cell(base, dataset, model):
    configs = []
    for ms in MSEEDS:
        path = f"{base}/{ms}/{dataset}/{model}/best_hyperparameters.yaml"
        if os.path.exists(path):
            with open(path) as f:
                configs.append(yaml.safe_load(f) or {})
    return configs


def mode_with_flag(values):
    """Return (mode_value, disagreed_bool). disagreed = any mseed differs from mode."""
    if not values:
        return None, False
    counts = Counter(str(v) for v in values)
    mode_str, _ = counts.most_common(1)[0]
    # recover original value that matches mode_str
    mode_val = next(v for v in values if str(v) == mode_str)
    disagreed = len(counts) > 1
    return mode_val, disagreed


def fmt_lr(v):
    if v is None:
        return "---"
    return f"${float(v):g}$"


def fmt_reg(v):
    if v is None:
        return "---"
    f = float(v)
    if f == 0:
        return "$0$"
    # express as 10^x
    import math

    exp = round(math.log10(f))
    if abs(f - 10**exp) / f < 1e-6:
        return f"$10^{{{exp}}}$"
    return f"${f:g}$"


def fmt_int(v):
    return "---" if v is None else f"${int(v)}$"


def fmt_plain(v):
    return "---" if v is None else f"${v:g}$" if isinstance(v, (int, float)) else f"{v}"


def dagger(s, flag):
    if not flag or s == "---":
        return s
    # if s is math-wrapped ("$...$"), brace-wrap the inner content so a
    # pre-existing superscript (e.g. "10^{-3}") doesn't collide with ^{\dagger}
    if s.startswith("$") and s.endswith("$"):
        inner = s[1:-1]
        return f"${{{inner}}}^{{\\dagger}}$"
    # plain numeric like "0.005" -> wrap in math mode with dagger
    return f"${s}^{{\\dagger}}$"


def compute_row(configs, keys):
    """For each key, return (mode_val, disagreed_flag)."""
    result = {}
    for k in keys:
        vals = [c.get(k) for c in configs if k in c]
        if not vals:
            result[k] = (None, False)
        else:
            result[k] = mode_with_flag(vals)
    return result


# hyperparameter keys per model (subset used in tables)
MODEL_KEYS = {
    "BPR": ["learning_rate", "weight_decay"],
    "LightGCN": ["learning_rate", "n_layers", "reg_weight"],
    "SimGCL": ["learning_rate", "n_layers", "reg_weight", "lambda", "eps", "temperature"],
    "DiffNet": ["learning_rate", "n_layers", "reg_weight"],
    "MHCN": ["learning_rate", "n_layers", "reg_weight", "ssl_reg"],
    "HAGN": ["learning_rate", "n_layers", "reg_weight"],
    "CSGCN": ["learning_rate", "n_layers", "reg_weight", "warmup_epochs", "anneal_epochs"],
    "ASLGCN": ["learning_rate", "n_layers", "reg_weight", "social_layers"],
}


def build_rows(base, dataset, models):
    out = {}
    for m in models:
        configs = load_cell(base, dataset, m)
        keys = MODEL_KEYS[m]
        out[m] = compute_row(configs, keys)
    return out


def render_cell(val, disagreed, kind="reg"):
    if val is None:
        return "---"
    if kind == "lr":
        s = fmt_lr(val)
    elif kind == "reg":
        s = fmt_reg(val)
    elif kind == "int":
        s = fmt_int(val)
    else:
        s = fmt_plain(val)
    return dagger(s, disagreed)


pass



def cell_lr(row, key="learning_rate"):
    v, d = row.get(key, (None, False))
    return render_cell(v, d, "lr")


def cell_int(row, key):
    v, d = row.get(key, (None, False))
    return render_cell(v, d, "int")


def cell_reg(row, key):
    v, d = row.get(key, (None, False))
    return render_cell(v, d, "reg")


def other_col(m, row):
    if m == "SimGCL":
        t, dt = row.get("temperature", (None, False))
        return "temp=" + render_cell(t, dt, "plain")
    if m == "HAGN":
        return "---"
    if m == "CSGCN":
        w, dw = row["warmup_epochs"]
        a, da = row["anneal_epochs"]
        return "warm=" + render_cell(w, dw, "int") + ", anneal=" + render_cell(a, da, "int")
    if m == "ASLGCN":
        sl, dsl = row["social_layers"]
        return "soc\\_L=" + render_cell(sl, dsl, "int")
    return "---"


def format_row(model, row):
    lr = cell_lr(row)
    layers = cell_int(row, "n_layers") if "n_layers" in MODEL_KEYS[model] else "---"
    reg = cell_reg(row, "reg_weight") if "reg_weight" in MODEL_KEYS[model] else "---"
    wd = cell_reg(row, "weight_decay") if "weight_decay" in MODEL_KEYS[model] else "---"
    lam = render_cell(*row["lambda"], "plain") if "lambda" in MODEL_KEYS[model] else "---"
    eps = render_cell(*row["eps"], "plain") if "eps" in MODEL_KEYS[model] else "---"
    ssl = cell_reg(row, "ssl_reg") if "ssl_reg" in MODEL_KEYS[model] else "---"
    other = other_col(model, row)
    return f"{model:10s}& {lr} & {layers} & {reg} & {wd} & {lam} & {eps} & {ssl} & {other}"


def format_hybrid_row(model, row):
    """Narrower row for the hybrid synthetic table: lr, layers, reg_wt, Other."""
    lr = cell_lr(row)
    layers = cell_int(row, "n_layers")
    reg = cell_reg(row, "reg_weight")
    other = other_col(model, row)
    return f"{model:10s}& {lr} & {layers} & {reg} & {other}"


print("\n% ── SYNTH ESTABLISHED table body (9 topologies x 5 established) ──")
for topo_label, topo_dataset in PRIMARY_TOPOS:
    rows = build_rows(SYNTH_BASE, topo_dataset, ESTABLISHED)
    print(f"% {topo_label}")
    for m in ESTABLISHED:
        print(f"  & {m} " + format_row(m, rows[m]).split("&", 1)[1] + r" \\")

print("\n% ── SYNTH HYBRID table body (9 topologies x 3 hybrids) ──")
for topo_label, topo_dataset in PRIMARY_TOPOS:
    rows = build_rows(SYNTH_BASE, topo_dataset, HYBRIDS)
    print(f"% {topo_label}")
    for m in HYBRIDS:
        print(f"  & {m} " + format_hybrid_row(m, rows[m]).split("&", 1)[1] + r" \\")

for ds in REAL_DATASETS:
    print(f"\n% ── REAL: {ds} ──")
    rows = build_rows(REAL_BASE, ds, MODELS)
    for m in MODELS:
        print(format_row(m, rows[m]) + r" \\")


def task_full_agreement(base, dataset, model):
    """A task (dataset,model) has full agreement if all 3 mseeds picked the same HP tuple."""
    configs = load_cell(base, dataset, model)
    if len(configs) < 3:
        return None
    keys = MODEL_KEYS[model]
    tuples = [tuple(c.get(k) for k in keys) for c in configs]
    return len(set(tuples)) == 1


def dagger_rate(base, datasets, models, skip_lr=False):
    total = 0
    dc = 0
    for ds in datasets:
        for m in models:
            rows = build_rows(base, ds, [m])[m]
            for k, (v, d) in rows.items():
                if skip_lr and k == "learning_rate":
                    continue
                if v is not None:
                    total += 1
                    dc += int(d)
    return dc, total


# synth established
est_agree = sum(
    1 for _, ds in PRIMARY_TOPOS for m in ESTABLISHED if task_full_agreement(SYNTH_BASE, ds, m)
)
est_total = len(PRIMARY_TOPOS) * len(ESTABLISHED)
hyb_agree = sum(
    1 for _, ds in PRIMARY_TOPOS for m in HYBRIDS if task_full_agreement(SYNTH_BASE, ds, m)
)
hyb_total = len(PRIMARY_TOPOS) * len(HYBRIDS)
real_agree = sum(1 for ds in REAL_DATASETS for m in MODELS if task_full_agreement(REAL_BASE, ds, m))
real_total = len(REAL_DATASETS) * len(MODELS)

print("\n% Full-agreement tasks:")
print(f"%   synth established: {est_agree}/{est_total} = {100 * est_agree / est_total:.0f}%")
print(f"%   synth hybrids:     {hyb_agree}/{hyb_total} = {100 * hyb_agree / hyb_total:.0f}%")
print(f"%   real:              {real_agree}/{real_total} = {100 * real_agree / real_total:.0f}%")

dc_est, tot_est = dagger_rate(SYNTH_BASE, [ds for _, ds in PRIMARY_TOPOS], ESTABLISHED)
dc_hyb, tot_hyb = dagger_rate(SYNTH_BASE, [ds for _, ds in PRIMARY_TOPOS], HYBRIDS)
dc_real, tot_real = dagger_rate(REAL_BASE, REAL_DATASETS, MODELS)
dc_all = dc_est + dc_hyb + dc_real
tot_all = tot_est + tot_hyb + tot_real
print("\n% Dagger rates:")
print(f"%   synth established: {dc_est}/{tot_est} = {100 * dc_est / tot_est:.0f}%")
print(f"%   synth hybrids:     {dc_hyb}/{tot_hyb} = {100 * dc_hyb / tot_hyb:.0f}%")
print(f"%   real:              {dc_real}/{tot_real} = {100 * dc_real / tot_real:.0f}%")
print(f"%   overall:           {dc_all}/{tot_all} = {100 * dc_all / tot_all:.0f}%")
