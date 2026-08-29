from __future__ import annotations

from ast import literal_eval
from pathlib import Path
from typing import Any, Dict
import ast
import colorsys

import matplotlib as mpl
import matplotlib.colors as mcolors


ROOT = Path(__file__).resolve().parents[1]
COLOR_SCHEME_CANDIDATES = [
    ROOT / "ressources" / "20260225_Color_scheme_Regional.txt",
    ROOT / "Ressources" / "20260225_Color_scheme_Regional.txt",
]

# Modality colours are sampled from the plasma colormap so that the modality
# legend sits in its own colour family, clearly separated from the viridis-based
# diet scheme. Each modality anchors a colour family; individual datasets are
# rendered as shades of their modality's anchor (see build_dataset_colors).
_PLASMA = mpl.colormaps["plasma"]

MODALITY_COLORS = {
    "proteomics": mcolors.to_hex(_PLASMA(0.5)),        # #cc4778 - red/purple
    "transcriptomics": mcolors.to_hex(_PLASMA(0.8)),   # #fca636 - yellow/orange
    # bulk_rnaseq intentionally omitted until the organoid bulk data is added;
    # it falls back to grey via get_modality_color().
}

# Hue / lightness / saturation ramps used to derive per-dataset shades within a
# modality. Hue is given in matplotlib HLS units (0-1).
#   transcriptomics: orange -> yellow
#   proteomics:      purple -> red
MODALITY_SHADE_RANGES = {
    "transcriptomics": {"hue": (0.055, 0.135), "light": (0.38, 0.72), "sat": (0.95, 0.85)},
    # starts at magenta rather than violet so the darkest proteomics shade stays
    # clear of the blue-purple HFDO diet colour (hue 0.74)
    "proteomics": {"hue": (0.84, 0.99), "light": (0.32, 0.70), "sat": (0.55, 0.75)},
}

_FALLBACK_SHADE_RANGE = {"hue": (0.55, 0.62), "light": (0.35, 0.70), "sat": (0.15, 0.25)}

# Intestinal segments in anatomical (proximal -> distal) order. Used both for
# categorical ordering of plot columns and for legend ordering.
SEGMENT_ORDER = ["Duodenum", "Jejunum", "Ileum", "Colon"]

# Diet groups in experimental order (control first).
DIET_ORDER = ["CD", "HFDL", "HFDO"]

# Fallback diet when a dataset carries no explicit diet annotation: all regional
# samples and the proteomics reference material are control diet.
DEFAULT_DIET = "CD"



def _candidate_color_file() -> Path:
    for candidate in COLOR_SCHEME_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not locate 20260225_Color_scheme_Regional.txt in ressources/")


def _load_assignments(path: Path) -> Dict[str, Any]:
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    out: Dict[str, Any] = {}
    for node in module.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            out[target.id] = literal_eval(node.value)
        except Exception:
            continue
    return out


def load_color_scheme() -> Dict[str, Dict[str, str]]:
    raw = _load_assignments(_candidate_color_file())
    return {
        "batch": dict(raw.get("batch", {})),
        "diet": dict(raw.get("diet", {})),
        "segment": dict(raw.get("segment", {})),
        "sample": dict(raw.get("sample", {})),
        "Manual_CT_lvl0": dict(raw.get("Manual_CT_lvl0", {})),
        "Manual_CT_lvl1": dict(raw.get("Manual_CT_lvl1", {})),
        "Manual_CT_lvl2": dict(raw.get("Manual_CT_lvl2", {})),
    }


def get_modality_color(modality: str) -> str:
    return MODALITY_COLORS.get(str(modality), "#666666")


def get_diet_color(diet: str, scheme: Dict[str, Dict[str, str]] | None = None) -> str:
    scheme = scheme or load_color_scheme()
    return scheme.get("diet", {}).get(str(diet), "#999999")


def get_segment_color(segment: str, scheme: Dict[str, Dict[str, str]] | None = None) -> str:
    scheme = scheme or load_color_scheme()
    return scheme.get("segment", {}).get(str(segment), "#999999")


def get_celltype_color(celltype: str, level: str = "Manual_CT_lvl1", scheme: Dict[str, Dict[str, str]] | None = None) -> str:
    scheme = scheme or load_color_scheme()
    return scheme.get(level, {}).get(str(celltype), "#999999")


def _shade(shade_range: Dict[str, tuple], position: float) -> str:
    """Interpolate one shade within a modality's hue/lightness/saturation ramp."""
    hue = shade_range["hue"][0] + (shade_range["hue"][1] - shade_range["hue"][0]) * position
    light = shade_range["light"][0] + (shade_range["light"][1] - shade_range["light"][0]) * position
    sat = shade_range["sat"][0] + (shade_range["sat"][1] - shade_range["sat"][0]) * position
    return mcolors.to_hex(colorsys.hls_to_rgb(hue % 1.0, light, sat))


def build_dataset_colors(modality_by_dataset: Dict[str, str]) -> Dict[str, str]:
    """Assign each dataset a shade of its modality's colour family.

    Datasets are grouped by modality and sorted by name, then spread evenly
    across that modality's hue/lightness ramp, so a glance at the dataset track
    shows both which modality a column belongs to and which dataset within it.
    Ordering is deterministic, so the same selection always yields the same
    colours across re-runs of the app.
    """
    by_modality: Dict[str, list] = {}
    for dataset, modality in modality_by_dataset.items():
        by_modality.setdefault(str(modality), []).append(str(dataset))

    colors: Dict[str, str] = {}
    for modality, datasets in by_modality.items():
        shade_range = MODALITY_SHADE_RANGES.get(modality, _FALLBACK_SHADE_RANGE)
        ordered = sorted(datasets)
        n = len(ordered)
        for index, dataset in enumerate(ordered):
            position = 0.5 if n == 1 else index / (n - 1)
            colors[dataset] = _shade(shade_range, position)
    return colors
