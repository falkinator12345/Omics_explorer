from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Set

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _resolve_level_column(manual_ct_lvl: str) -> str:
    if manual_ct_lvl.startswith("Manual_CT_"):
        return manual_ct_lvl
    return f"Manual_CT_{manual_ct_lvl}"


def load_aggregated_expression(
    csv_path: str | Path,
    manual_ct_lvl: str = "lvl1",
    genes: Iterable[str] | None = None,
    max_rows: int | None = None,
) -> pd.DataFrame:
    level_col = _resolve_level_column(manual_ct_lvl)
    header = pd.read_csv(csv_path, nrows=0).columns.tolist()

    required_base = {"gene", "mean", "pct_expr"}
    missing_base = sorted(required_base.difference(header))
    if missing_base:
        raise ValueError(
            f"Missing required columns in aggregated expression CSV: {missing_base}"
        )
    if level_col not in header:
        raise ValueError(
            f"Expected level column '{level_col}' not found in aggregated expression CSV."
        )

    optional_segment_col = None
    for candidate in ["segment", "Segment", "intestinal_segment", "Intestinal_segment"]:
        if candidate in header:
            optional_segment_col = candidate
            break

    usecols = ["gene", level_col, "mean", "pct_expr"]
    if optional_segment_col:
        usecols.append(optional_segment_col)

    expr_df = pd.read_csv(
        csv_path,
        usecols=usecols,
        nrows=max_rows,
    ).rename(columns={level_col: "celltype_label"})

    if optional_segment_col:
        expr_df = expr_df.rename(columns={optional_segment_col: "segment_label"})

    if genes is not None:
        wanted = {str(gene).casefold() for gene in genes}
        expr_df = expr_df[
            expr_df["gene"].astype(str).str.casefold().isin(wanted)
        ].copy()

    expr_df["gene"] = expr_df["gene"].astype(str)
    expr_df["celltype_label"] = expr_df["celltype_label"].astype(str)
    expr_df["mean"] = pd.to_numeric(expr_df["mean"], errors="coerce")
    expr_df["pct_expr"] = pd.to_numeric(expr_df["pct_expr"], errors="coerce")

    out_cols = ["gene", "celltype_label", "mean", "pct_expr"]
    if "segment_label" in expr_df.columns:
        out_cols.append("segment_label")
    return expr_df[out_cols]


def plot_aggregated_heatmap(
    expr_df: pd.DataFrame,
    value_col: str = "mean",
    top_n_genes: int = 30,
    title: str = "Aggregated Expression Heatmap",
    output_path: str | Path | None = None,
):
    required = {"gene", "celltype_label", value_col}
    missing = sorted(required.difference(expr_df.columns))
    if missing:
        raise ValueError(f"Missing required columns for heatmap: {missing}")

    heat = expr_df.pivot_table(
        index="gene",
        columns="celltype_label",
        values=value_col,
        aggfunc="mean",
    )
    if heat.empty:
        raise ValueError("No data available to plot aggregated heatmap.")

    if top_n_genes and heat.shape[0] > top_n_genes:
        top_genes = heat.mean(axis=1).sort_values(ascending=False).head(top_n_genes).index
        heat = heat.loc[top_genes]

    fig, ax = plt.subplots(
        figsize=(max(8, 0.45 * heat.shape[1]), max(6, 0.3 * heat.shape[0]))
    )
    image = ax.imshow(heat.fillna(0).to_numpy(), aspect="auto", cmap="viridis")

    ax.set_title(title)
    ax.set_xlabel("Cell Type")
    ax.set_ylabel("Gene")
    ax.set_xticks(range(len(heat.columns)))
    ax.set_xticklabels(heat.columns, rotation=90)
    ax.set_yticks(range(len(heat.index)))
    ax.set_yticklabels(heat.index)

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(value_col)
    fig.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")

    return fig


def plot_aggregated_dotplot(
    expr_df: pd.DataFrame,
    title: str = "Aggregated Expression Dotplot",
    output_path: str | Path | None = None,
):
    required = {"gene", "celltype_label", "mean", "pct_expr"}
    missing = sorted(required.difference(expr_df.columns))
    if missing:
        raise ValueError(f"Missing required columns for dotplot: {missing}")

    plot_df = (
        expr_df[["gene", "celltype_label", "mean", "pct_expr"]]
        .groupby(["gene", "celltype_label"], as_index=False)
        .mean(numeric_only=True)
    )
    if plot_df.empty:
        raise ValueError("No data available to plot aggregated dotplot.")

    x_labels = list(pd.unique(plot_df["celltype_label"]))
    y_labels = list(pd.unique(plot_df["gene"]))
    x_map = {label: index for index, label in enumerate(x_labels)}
    y_map = {label: index for index, label in enumerate(y_labels)}

    pct = plot_df["pct_expr"].fillna(0).clip(lower=0)
    max_pct = float(pct.max()) if len(pct) else 0.0
    sizes = 40 + 260 * (pct / max_pct) if max_pct > 0 else pd.Series(40, index=plot_df.index)

    fig, ax = plt.subplots(
        figsize=(max(8, 0.45 * len(x_labels)), max(6, 0.3 * len(y_labels)))
    )
    scatter = ax.scatter(
        plot_df["celltype_label"].map(x_map).to_numpy(),
        plot_df["gene"].map(y_map).to_numpy(),
        c=plot_df["mean"],
        s=sizes,
        cmap="viridis",
        alpha=0.9,
        edgecolors="none",
    )

    ax.set_title(title)
    ax.set_xlabel("Cell Type")
    ax.set_ylabel("Gene")
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=90)
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels)

    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("mean")
    fig.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")

    return fig


def _shorten_label(label: str, max_chars: int = 55) -> str:
    if len(label) <= max_chars:
        return label
    head = max_chars // 2 - 1
    tail = max_chars - head - 1
    return f"{label[:head]}…{label[-tail:]}"


def plot_overlap(
    overlap_sets: Dict[str, Set[str]],
    title: str = "Feature Overlap",
    log_scale: bool = False,
    output_path: str | Path | None = None,
    intersection_order: list[str] | None = None,
    max_subsets: int = 40,
):
    """UpSet-style overlap plot.

    If ``intersection_order`` is given (list of exclusive-membership labels,
    each formatted as " ∩ ".join(sorted(set_names))), bars are ordered and
    numbered according to their position in that list, so the plotted
    intersection IDs match the summary table / dropdown in the app.
    """
    n = len(overlap_sets)
    if n < 2:
        raise ValueError("Need at least 2 comparison sets to plot overlap.")

    labels = list(overlap_sets.keys())
    label_count = len(labels)

    memberships: dict[tuple[bool, ...], int] = {}
    universe = sorted(set().union(*overlap_sets.values()))
    for gene in universe:
        mask = tuple(gene in overlap_sets[label] for label in labels)
        if not any(mask):
            continue
        memberships[mask] = memberships.get(mask, 0) + 1

    if not memberships:
        raise ValueError("No intersections available for the selected sets.")

    def _mask_label(mask: tuple[bool, ...]) -> str:
        return " ∩ ".join(sorted(label for label, present in zip(labels, mask) if present))

    multi_subsets = [(mask, size) for mask, size in memberships.items() if sum(mask) >= 2]
    if not multi_subsets:
        raise ValueError("No multi-set intersections available for the selected sets.")

    if intersection_order:
        id_by_label = {label: index for index, label in enumerate(intersection_order)}
        annotated = [
            (id_by_label.get(_mask_label(mask), len(id_by_label) + i), mask, size)
            for i, (mask, size) in enumerate(multi_subsets)
        ]
        annotated.sort(key=lambda item: item[0])
        sorted_subsets = [(mask, size) for _, mask, size in annotated]
        subset_ids = [order_index + 1 for order_index, _, _ in annotated]
    else:
        sorted_subsets = sorted(multi_subsets, key=lambda item: item[1], reverse=True)
        subset_ids = list(range(1, len(sorted_subsets) + 1))

    truncated = len(sorted_subsets) > max_subsets
    sorted_subsets = sorted_subsets[:max_subsets]
    subset_ids = subset_ids[:max_subsets]

    subset_masks = [mask for mask, _ in sorted_subsets]
    subset_sizes = [size for _, size in sorted_subsets]
    bar_heights = np.log2(np.asarray(subset_sizes, dtype=float) + 1.0) if log_scale else np.asarray(subset_sizes, dtype=float)
    x_positions = list(range(len(sorted_subsets)))

    n_bars = len(sorted_subsets)
    display_labels = [_shorten_label(label) for label in labels]
    label_pad_inches = min(6.0, 0.085 * max((len(lbl) for lbl in display_labels), default=10))
    fig_width = min(24.0, max(8.0, 0.42 * n_bars + label_pad_inches))
    fig_height = min(18.0, max(5.0, 0.32 * label_count + 3.5))
    tick_fontsize = 9 if n_bars <= 25 else 7
    matrix_fontsize = 9 if label_count <= 15 else 7

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, max(1.2, 0.28 * label_count)], hspace=0.06)

    ax_top = fig.add_subplot(gs[0])
    ax_mat = fig.add_subplot(gs[1], sharex=ax_top)

    bars = ax_top.bar(x_positions, bar_heights, color="#4c78a8", width=0.8)
    ax_top.set_ylabel("log2(intersection size + 1)" if log_scale else "Intersection size")
    ax_top.set_title(title + (f" (top {max_subsets} intersections)" if truncated else ""))
    ax_top.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

    for bar, raw_size in zip(bars, subset_sizes):
        height = bar.get_height()
        ax_top.annotate(
            str(raw_size),
            (bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=tick_fontsize - 1,
        )

    if len(bar_heights):
        ax_top.set_ylim(0, float(bar_heights.max()) * 1.2 + 0.1)

    empty_labels = [label for label, values in overlap_sets.items() if not values]
    if empty_labels:
        ax_top.text(
            0.99,
            0.98,
            f"Empty sets: {len(empty_labels)}",
            transform=ax_top.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "#f2f2f2",
                "edgecolor": "#bdbdbd",
            },
        )

    y_positions = list(range(label_count))
    for y, label in enumerate(labels):
        if label in empty_labels:
            ax_mat.axhspan(y - 0.45, y + 0.45, color="#f2f2f2", zorder=0)

    dot_size = 30 if n_bars <= 25 else 18
    for x, mask in zip(x_positions, subset_masks):
        present_rows = []
        for y, present in enumerate(mask):
            ax_mat.scatter(x, y, s=dot_size, color="#d9d9d9", zorder=1)
            if present:
                present_rows.append(y)
                ax_mat.scatter(x, y, s=dot_size + 5, color="#222222", zorder=2)
        if len(present_rows) > 1:
            ax_mat.plot([x, x], [min(present_rows), max(present_rows)], color="#222222", linewidth=1.2, zorder=1)

    ax_mat.set_yticks(y_positions)
    ax_mat.set_yticklabels(display_labels, fontsize=matrix_fontsize)
    ax_mat.set_ylim(-0.5, label_count - 0.5)
    ax_mat.invert_yaxis()
    ax_mat.set_xlabel("Intersection ID" if intersection_order else "Intersections (sorted by size)")
    ax_mat.set_xticks(x_positions)
    ax_mat.set_xticklabels([str(subset_id) for subset_id in subset_ids], rotation=0, fontsize=tick_fontsize)
    ax_mat.grid(axis="x", linestyle=":", alpha=0.2)

    for tick_label, label in zip(ax_mat.get_yticklabels(), labels):
        if label in empty_labels:
            tick_label.set_color("#7f7f7f")

    max_label_len = max((len(lbl) for lbl in display_labels), default=10)
    left_margin = min(0.48, 0.04 + 0.0065 * max_label_len)
    fig.subplots_adjust(left=left_margin, right=0.98, bottom=0.09, top=0.92)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")

    return fig
