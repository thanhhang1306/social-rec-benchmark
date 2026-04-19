"""Shared color palette for thesis figures (Okabe-Ito + viridis)."""

_OI = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
    "grey": "#7F7F7F",
}

# shared across all thesis figure scripts
MODEL_COLORS = {
    "BPR": _OI["grey"],
    "LightGCN": _OI["blue"],
    "SimGCL": _OI["reddish_purple"],
    "DiffNet": _OI["vermillion"],
    "MHCN": _OI["bluish_green"],
    "HAGN": _OI["orange"],
    "CSGCN": _OI["sky_blue"],
    "ASLGCN": _OI["yellow"],
}

GROUP_COLORS = {
    "I": _OI["blue"],
    "II": _OI["orange"],
    "III": _OI["bluish_green"],
}

SEQUENTIAL_CMAP = "viridis"

ACCENT = _OI["vermillion"]
MUTED = "#888888"
