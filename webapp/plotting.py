from __future__ import annotations

import re

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from webapp.colors import (
    DEFAULT_DIET,
    DIET_ORDER,
    SEGMENT_ORDER,
    build_dataset_colors,
    get_celltype_color,
    get_diet_color,
    get_modality_color,
    get_segment_color,
    load_color_scheme,
)


COLOR_SCHEME = load_color_scheme()

# Streamlit rasterizes matplotlib figures to a fixed 200 dpi PNG before
# stretching them to the container width (see st.pyplot's internal
# marshall()). A hardcoded dpi looks soft on wide/high-DPI screens for our
# narrower figures and wastes memory upscaling dpi on our widest ones, so we
# pick a dpi per-figure that targets a roughly constant rendered pixel width
# instead of a constant dpi.
_RENDER_TARGET_PX = 2200
_RENDER_MIN_DPI = 130
_RENDER_MAX_DPI = 260


def render_dpi(fig) -> int:
    """Pick a savefig dpi that keeps `fig` sharp when stretched to fill the
    browser without over-rendering (and thus over-allocating memory for)
    already-wide figures.
    """
    width_in, _height_in = fig.get_size_inches()
    dpi = _RENDER_TARGET_PX / max(width_in, 1.0)
    return int(min(_RENDER_MAX_DPI, max(_RENDER_MIN_DPI, dpi)))

_SEGMENT_RANK = {segment.casefold(): index for index, segment in enumerate(SEGMENT_ORDER)}
_DIET_RANK = {diet.casefold(): index for index, diet in enumerate(DIET_ORDER)}


def _token_sort_value(token: str) -> tuple[int, int, str]:
    """Rank a single label token: segments proximal->distal, diets control-first."""
    key = str(token).strip().casefold()
    if key in _SEGMENT_RANK:
        return (0, _SEGMENT_RANK[key], "")
    if key in _DIET_RANK:
        return (1, _DIET_RANK[key], "")
    return (2, 0, key)


def column_sort_key(label: str) -> tuple:
    """Sort key for column labels such as 'Colon | Absorptive' or 'Absorptive | CD'.

    Tokens are ranked positionally, so whichever field comes first in the label
    (celltype-first or condition-first grouping) drives the primary ordering,
    while segments always appear proximal -> distal and diets control-first.
    """
    parts = [part.strip() for part in str(label).split("|")]
    return tuple(_token_sort_value(part) for part in parts)


def order_columns(labels) -> list[str]:
    return sorted({str(label) for label in labels}, key=column_sort_key)


def _column_meta(expr_df: pd.DataFrame, column_key: str, value_col: str) -> pd.DataFrame:
    keep_cols = [
        col
        for col in [
            "modality",
            "celltype_label",
            "manual_ct_lvl",
            "sample_class",
            "source_dataset",
            "group_label",
            "dataset_label",
        ]
        if col in expr_df.columns and col != column_key
    ]
    meta = expr_df[[column_key] + keep_cols].drop_duplicates(subset=[column_key]).copy()
    return meta


def _label_tokens(row: pd.Series) -> set[str]:
    tokens: list[str] = []
    for column in ["group_label", "sample_class", "dataset_label", "celltype_label"]:
        value = row.get(column)
        if value is not None and pd.notna(value):
            tokens.extend(re.split(r"[|_\s]+", str(value)))
    return {token.strip().upper() for token in tokens if token.strip()}


def resolve_diet(row: pd.Series) -> str:
    """Diet for a column; defaults to CD when no diet is stated explicitly."""
    tokens = _label_tokens(row)
    for diet in reversed(DIET_ORDER):  # match HFDO / HFDL before CD
        if diet.upper() in tokens:
            return diet
    return DEFAULT_DIET


def resolve_segment(row: pd.Series) -> str | None:
    tokens = _label_tokens(row)
    for segment in SEGMENT_ORDER:
        if segment.upper() in tokens:
            return segment
    return None


def _annotation_frame(meta: pd.DataFrame, column_key: str, columns: list[str]) -> pd.DataFrame:
    """Per-column diet / dataset / modality annotation values, in plot order."""
    indexed = meta.drop_duplicates(subset=[column_key]).set_index(column_key)
    rows = []
    for column in columns:
        row = indexed.loc[column] if column in indexed.index else pd.Series(dtype=object)
        dataset = row.get("dataset_label")
        if dataset is None or (isinstance(dataset, float) and pd.isna(dataset)):
            dataset = row.get("source_dataset", "unknown")
        rows.append(
            {
                column_key: column,
                "diet": resolve_diet(row),
                "dataset": str(dataset),
                "modality": str(row.get("modality", "unknown")),
            }
        )
    return pd.DataFrame(rows)


def _annotation_tracks(ann: pd.DataFrame, dataset_colors: dict[str, str]):
    """Build (track name, per-column colours, legend mapping), ordered bottom-up."""
    diet_map = {diet: get_diet_color(diet, COLOR_SCHEME) for diet in ann["diet"].unique()}
    dataset_map = {name: dataset_colors.get(name, "#999999") for name in ann["dataset"].unique()}
    modality_map = {modality: get_modality_color(modality) for modality in ann["modality"].unique()}
    return [
        ("diet", [diet_map[value] for value in ann["diet"]], diet_map),
        ("dataset", [dataset_map[value] for value in ann["dataset"]], dataset_map),
        ("modality", [modality_map[value] for value in ann["modality"]], modality_map),
    ]


def _draw_annotation_tracks(fig, ax, tracks, bar_height: float = 0.022, gap: float = 0.004) -> None:
    """Draw stacked colour strips above `ax`, one per annotation level.

    Must be called AFTER layout adjustment (subplots_adjust / tight_layout),
    otherwise the strips anchor to stale axes positions and end up off-canvas.
    Tracks stack upwards in the order given (diet closest to the plot).
    """
    pos = ax.get_position()
    for index, (name, colors, _legend) in enumerate(tracks):
        if not colors:
            continue
        y0 = pos.y1 + gap + index * (bar_height + gap)
        if y0 + bar_height > 1.0:
            break
        bar_ax = fig.add_axes([pos.x0, y0, pos.width, bar_height])
        arr = np.array([list(range(len(colors)))], dtype=float)
        cmap = plt.matplotlib.colors.ListedColormap(colors)
        bar_ax.imshow(arr, aspect="auto", cmap=cmap, interpolation="nearest")
        bar_ax.set_xticks([])
        bar_ax.set_yticks([])
        bar_ax.set_ylabel(name, rotation=0, labelpad=8, va="center", ha="right", fontsize=7)
        for spine in bar_ax.spines.values():
            spine.set_visible(False)


def _legend_label(name: str, value: str, max_chars: int = 34) -> str:
    text = str(value)
    if name == "dataset":
        text = text.replace("proteomics:", "").replace("_Mixed_SE_marked", "").replace("_SE_marked", "")
    if len(text) > max_chars:
        text = text[: max_chars - 1] + "…"
    return text


def _sorted_legend_items(name: str, mapping: dict[str, str]) -> list[tuple[str, str]]:
    if name == "diet":
        order = [d for d in DIET_ORDER if d in mapping] + sorted(set(mapping) - set(DIET_ORDER))
    else:
        order = sorted(mapping)
    return [(_legend_label(name, key), mapping[key]) for key in order]


def _add_annotation_legends(fig, tracks) -> None:
    """One legend per annotation level, laid out left-to-right above the plot."""
    active = [(name, mapping) for name, _colors, mapping in reversed(tracks) if mapping]
    if not active:
        return
    x = 0.01
    for name, mapping in active:
        items = _sorted_legend_items(name, mapping)
        handles = [mpatches.Patch(color=color, label=label) for label, color in items]
        ncol = 2 if len(handles) > 5 else 1
        fig.legend(
            handles=handles,
            title=name,
            loc="upper left",
            bbox_to_anchor=(x, 1.0),
            ncol=ncol,
            frameon=False,
            fontsize=6.5,
            title_fontsize=7,
            handlelength=1.0,
            handleheight=1.0,
            labelspacing=0.25,
            borderpad=0.2,
            alignment="left",
        )
        # advance by longest label so legends do not overlap
        longest = max((len(str(label)) for label, _ in items), default=6)
        x += min(0.46, 0.035 + 0.0075 * longest) * (1.7 if ncol > 1 else 1.0)


def _annotate(fig, ax, meta: pd.DataFrame, column_key: str, columns: list[str], dataset_colors: dict[str, str], legends: bool = True) -> None:
    ann = _annotation_frame(meta, column_key, columns)
    tracks = _annotation_tracks(ann, dataset_colors)
    _draw_annotation_tracks(fig, ax, tracks)
    if legends:
        _add_annotation_legends(fig, tracks)


def _modality_by_dataset(expr_df: pd.DataFrame) -> dict[str, str]:
    if "dataset_label" not in expr_df.columns:
        return {}
    frame = expr_df.drop_duplicates(subset=["dataset_label"])
    modality = frame["modality"].astype(str) if "modality" in frame.columns else "unknown"
    return dict(zip(frame["dataset_label"].astype(str), modality))


def _short_column_label(label: str) -> str:
    """Drop the dataset prefix from combined-plot tick labels.

    The dataset of origin is already shown by the `dataset` annotation track,
    so repeating it on every tick just makes the axis unreadable.
    """
    parts = [part.strip() for part in str(label).split("|")]
    return " | ".join(parts[1:]) if len(parts) > 1 else str(label)


def _combined_figsize(n_cols: int, n_rows: int) -> tuple[float, float]:
    width = min(20.0, max(6.5, 0.3 * n_cols + 3.0))
    height = min(14.0, max(4.2, 0.25 * n_rows + 3.0))
    return width, height


def heatmap_combined(expr_df: pd.DataFrame, value_col: str = "z_value", title: str = "Combined Heatmap"):
    if expr_df.empty:
        raise ValueError("No expression data available for plotting.")

    plot_df = expr_df.copy()
    plot_df["axis_group"] = plot_df["dataset_label"] + " | " + plot_df["group_label"].astype(str)
    meta = _column_meta(plot_df, "axis_group", value_col)

    matrix = plot_df.pivot_table(
        index="gene",
        columns="axis_group",
        values=value_col,
        aggfunc="mean",
    ).fillna(0)
    matrix = matrix[order_columns(matrix.columns)]

    columns = list(matrix.columns)
    dataset_colors = build_dataset_colors(_modality_by_dataset(plot_df))
    tick_fs = 8 if len(columns) <= 40 else 6
    fig, ax = plt.subplots(figsize=_combined_figsize(len(columns), len(matrix.index)))
    im = ax.imshow(matrix.to_numpy(), aspect="auto", cmap="coolwarm", vmin=-2.5, vmax=2.5)
    ax.set_xlabel("Group (dataset shown in annotation track)")
    ax.set_ylabel("Gene")
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels([_short_column_label(c) for c in columns], rotation=90, fontsize=tick_fs)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index, fontsize=tick_fs)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(value_col)

    # Layout first, THEN annotation tracks (they anchor to the final axes position).
    fig.subplots_adjust(left=0.16, right=0.92, bottom=0.32, top=0.68)
    fig.suptitle(title, fontsize=10, y=0.88)
    _annotate(fig, ax, meta, "axis_group", columns, dataset_colors)
    return fig


def dotplot_combined(expr_df: pd.DataFrame, value_col: str = "z_value", title: str = "Combined Dotplot"):
    if expr_df.empty:
        raise ValueError("No expression data available for plotting.")

    plot_df = expr_df.copy()
    plot_df["axis_group"] = plot_df["dataset_label"] + " | " + plot_df["group_label"].astype(str)
    meta = _column_meta(plot_df, "axis_group", value_col)
    agg = (
        plot_df.groupby(["gene", "axis_group"], as_index=False)[value_col]
        .mean(numeric_only=True)
        .fillna(0)
    )

    x_labels = order_columns(agg["axis_group"])
    y_labels = list(pd.unique(agg["gene"]))
    x_map = {label: i for i, label in enumerate(x_labels)}
    y_map = {label: i for i, label in enumerate(y_labels)}

    size = (agg[value_col].abs() + 0.2) * 90

    dataset_colors = build_dataset_colors(_modality_by_dataset(plot_df))
    tick_fs = 8 if len(x_labels) <= 40 else 6
    fig, ax = plt.subplots(figsize=_combined_figsize(len(x_labels), len(y_labels)))
    sc = ax.scatter(
        agg["axis_group"].map(x_map),
        agg["gene"].map(y_map),
        c=agg[value_col],
        s=size,
        cmap="coolwarm",
        vmin=-2.5,
        vmax=2.5,
        alpha=0.9,
        edgecolors="none",
    )

    ax.set_xlabel("Group (dataset shown in annotation track)")
    ax.set_ylabel("Gene")
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels([_short_column_label(c) for c in x_labels], rotation=90, fontsize=tick_fs)
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=tick_fs)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(value_col)

    fig.subplots_adjust(left=0.16, right=0.92, bottom=0.32, top=0.68)
    fig.suptitle(title, fontsize=10, y=0.88)
    _annotate(fig, ax, meta, "axis_group", x_labels, dataset_colors)
    return fig


def heatmap_faceted(expr_df: pd.DataFrame, value_col: str = "z_value", title: str = "Faceted Heatmap"):
    datasets = list(pd.unique(expr_df["dataset_label"]))
    if not datasets:
        raise ValueError("No expression data available for plotting.")

    n = len(datasets)
    max_cols = max(expr_df.groupby("dataset_label")["group_label"].nunique())
    fig_width = min(18.0, max(7.0, 0.3 * max_cols + 3.0))
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(fig_width, max(3.2 * n, 5.0)))
    if n == 1:
        axes = [axes]

    dataset_colors = build_dataset_colors(_modality_by_dataset(expr_df))
    bar_jobs = []
    for ax, dataset in zip(axes, datasets):
        sub = expr_df[expr_df["dataset_label"] == dataset].copy()
        meta = _column_meta(sub, "group_label", value_col)
        matrix = sub.pivot_table(
            index="gene",
            columns="group_label",
            values=value_col,
            aggfunc="mean",
        ).fillna(0)
        matrix = matrix[order_columns(matrix.columns)]

        tick_fs = 8 if len(matrix.columns) <= 40 else 6
        im = ax.imshow(matrix.to_numpy(), aspect="auto", cmap="coolwarm", vmin=-2.5, vmax=2.5)
        ax.set_title(dataset, fontsize=9, pad=18)
        ax.set_xlabel("Group")
        ax.set_ylabel("Gene")
        ax.set_xticks(range(len(matrix.columns)))
        ax.set_xticklabels(matrix.columns, rotation=90, fontsize=tick_fs)
        ax.set_yticks(range(len(matrix.index)))
        ax.set_yticklabels(matrix.index, fontsize=tick_fs)
        fig.colorbar(im, ax=ax)
        bar_jobs.append((ax, meta, list(matrix.columns)))

    fig.tight_layout(rect=[0, 0, 1, 0.86])
    fig.suptitle(title, fontsize=10, y=0.95)
    # Annotation tracks must be added after tight_layout so they track final axes positions.
    for index, (ax, meta, columns) in enumerate(bar_jobs):
        _annotate(fig, ax, meta, "group_label", columns, dataset_colors, legends=(index == 0))
    return fig


def dotplot_faceted(expr_df: pd.DataFrame, value_col: str = "z_value", title: str = "Faceted Dotplot"):
    datasets = list(pd.unique(expr_df["dataset_label"]))
    if not datasets:
        raise ValueError("No expression data available for plotting.")

    n = len(datasets)
    max_cols = max(expr_df.groupby("dataset_label")["group_label"].nunique())
    fig_width = min(18.0, max(7.0, 0.3 * max_cols + 3.0))
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(fig_width, max(3.2 * n, 5.0)))
    if n == 1:
        axes = [axes]

    dataset_colors = build_dataset_colors(_modality_by_dataset(expr_df))
    bar_jobs = []
    for ax, dataset in zip(axes, datasets):
        sub = expr_df[expr_df["dataset_label"] == dataset].copy()
        meta = _column_meta(sub, "group_label", value_col)
        agg = (
            sub.groupby(["gene", "group_label"], as_index=False)[value_col]
            .mean(numeric_only=True)
            .fillna(0)
        )

        x_labels = order_columns(agg["group_label"])
        y_labels = list(pd.unique(agg["gene"]))
        x_map = {label: i for i, label in enumerate(x_labels)}
        y_map = {label: i for i, label in enumerate(y_labels)}
        size = (agg[value_col].abs() + 0.2) * 90

        tick_fs = 8 if len(x_labels) <= 40 else 6
        sc = ax.scatter(
            agg["group_label"].map(x_map),
            agg["gene"].map(y_map),
            c=agg[value_col],
            s=size,
            cmap="coolwarm",
            vmin=-2.5,
            vmax=2.5,
            alpha=0.9,
            edgecolors="none",
        )
        ax.set_title(dataset, fontsize=9, pad=18)
        ax.set_xlabel("Group")
        ax.set_ylabel("Gene")
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=90, fontsize=tick_fs)
        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(y_labels, fontsize=tick_fs)
        fig.colorbar(sc, ax=ax)
        bar_jobs.append((ax, meta, x_labels))

    fig.tight_layout(rect=[0, 0, 1, 0.86])
    fig.suptitle(title, fontsize=10, y=0.95)
    for index, (ax, meta, columns) in enumerate(bar_jobs):
        _annotate(fig, ax, meta, "group_label", columns, dataset_colors, legends=(index == 0))
    return fig
