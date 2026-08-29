from __future__ import annotations

from pathlib import Path
import re
from collections import defaultdict

import pandas as pd
import streamlit as st

from omics_overview.parsers import infer_sample_class_from_name
from omics_overview.visualize import plot_overlap
from webapp.data_layer import (
    describe_value_transform,
    gene_sets_for_comparisons,
    load_hfd_expression,
    load_proteomics_expression,
    load_regional_segment_expression,
    load_significance_table,
    load_snrna_expression,
    parse_gene_text,
    query_genes,
    zscore_within_dataset,
)
from webapp.plotting import (
    dotplot_combined,
    dotplot_faceted,
    heatmap_combined,
    heatmap_faceted,
)

st.set_page_config(page_title="Omics Local Explorer", layout="wide")

ROOT = Path(__file__).resolve().parent
PARQUET_PATH = ROOT / "consolidated_significant_findings.parquet"
DATA_DIR = ROOT / "Data"
SNRNA_AGG_CSV = DATA_DIR / "20260405_Regional_snRNAseq_aggregated_expression_per_celltype_long_format.csv"
HFD_AGG_PARQUET = DATA_DIR / "20260811_HFD_snRNAseq_aggregated_expression_per_celltype_long_format.parquet"
REGIONAL_SEG_PARQUET = DATA_DIR / "20260811_Regional_snRNAseq_aggregated_expression_per_celltype_segment_long_format.parquet"
PROTEOMICS_DIR = DATA_DIR / "Proteomics"


@st.cache_data(show_spinner=False)
def _load_significance(path: str) -> pd.DataFrame:
    return load_significance_table(path)


@st.cache_data(show_spinner=False)
def _load_proteomics_expr(proteomics_dir: str, genes_key: tuple[str, ...]) -> pd.DataFrame:
    return load_proteomics_expression(proteomics_dir, list(genes_key))


def _slug_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _friendly_proteomics_label(source_file: str) -> str:
    label = infer_sample_class_from_name(Path(source_file).name)
    if label and label != "Unknown":
        return label
    return Path(source_file).stem.replace("_", " ")


def _friendly_proteomics_label_from_stem(stem: str) -> str:
    label = infer_sample_class_from_name(stem)
    if label and label != "Unknown":
        return label
    return stem.replace("_", " ")


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _render_checkbox_choices(
    title: str,
    options: list[tuple[str, str]],
    key_prefix: str,
    default_selected: list[str] | None = None,
    columns: int = 2,
    disabled_values: set[str] | None = None,
) -> list[str]:
    if not options:
        return []

    defaults = set(default_selected or [])
    disabled = disabled_values or set()
    selected: list[str] = []
    with st.expander(title, expanded=False):
        column_count = max(1, min(columns, len(options)))
        grid = st.columns(column_count)
        for index, (value, label) in enumerate(options):
            key = f"{key_prefix}:{_slug_key(value)}"
            if value in disabled:
                st.session_state[key] = False
            elif key not in st.session_state:
                st.session_state[key] = value in defaults and value not in disabled
            with grid[index % column_count]:
                st.checkbox(label, key=key, disabled=value in disabled)
            if st.session_state[key]:
                selected.append(value)
    return selected


def _proteomics_dataset_stems() -> list[str]:
    stems = []
    for cache_file in sorted(PROTEOMICS_DIR.glob("*_expression_cache.csv")):
        stem = cache_file.stem
        if stem.endswith("_expression_cache"):
            stem = stem[: -len("_expression_cache")]
        stems.append(stem)
    return stems


def _expression_dataset_options() -> list[tuple[str, str]]:
    options: list[tuple[str, str]] = [("regional_snrnaseq", "Regional snRNAseq")]
    if HFD_AGG_PARQUET.exists():
        options.append(("hfd_snrnaseq:Colon", "HFD snRNAseq Colon (per diet)"))
        options.append(("hfd_snrnaseq:Jejunum", "HFD snRNAseq Jejunum (per diet)"))
    stems = _proteomics_dataset_stems()
    base_labels = [_friendly_proteomics_label_from_stem(stem) for stem in stems]
    label_counts = pd.Series(base_labels).value_counts().to_dict() if base_labels else {}
    for stem in stems:
        internal = f"proteomics:{stem}"
        base_label = _friendly_proteomics_label_from_stem(stem)
        label = base_label if label_counts.get(base_label, 0) <= 1 else f"{base_label} ({stem})"
        options.append((internal, label))
    return options


def _selection_signature(
    selected_comparisons: list[str],
    prot_selected: list[str],
    tx_selected: list[str],
    include_combined_prot: bool,
    include_combined_tx: bool,
) -> str:
    parts = [
        ",".join(sorted(selected_comparisons)),
        ",".join(sorted(prot_selected)),
        ",".join(sorted(tx_selected)),
        f"prot={int(include_combined_prot)}",
        f"tx={int(include_combined_tx)}",
    ]
    return "||".join(parts)


def _plot_selection_signature(
    source_mode: str,
    selected_genes: list[str],
    dataset_sel: list[str],
    manual_lvl: str,
    layout_mode: str,
    plot_type: str,
    split_by_segment: bool,
    condition_first: bool,
) -> str:
    parts = [
        source_mode,
        ",".join(selected_genes),
        ",".join(sorted(dataset_sel)),
        manual_lvl,
        layout_mode,
        plot_type,
        f"seg={int(split_by_segment)}",
        f"condfirst={int(condition_first)}",
    ]
    return "||".join(parts)


def _clear_state_prefix(prefix: str) -> None:
    for key in list(st.session_state.keys()):
        if key.startswith(prefix):
            del st.session_state[key]


def _transcriptomics_source_options(tx_df: pd.DataFrame) -> list[tuple[str, str, bool]]:
    labels = {
        "regional_snrnaseq": "Regional snRNAseq",
        "hfd_snrnaseq": "HFD snRNAseq",
        "bulk_rnaseq": "Bulk RNAseq",
        "organoid_bulk_rnaseq": "Organoid bulk RNAseq",
    }
    available_sources = set(tx_df["source_dataset"].dropna().astype(str).unique().tolist())
    out: list[tuple[str, str, bool]] = []
    for source_dataset, label in labels.items():
        out.append((source_dataset, label, source_dataset in available_sources))
    return out


def _parse_proteomics_selection(selected: list[str]) -> list[str]:
    return [token.split(":", 1)[1] for token in selected if token.startswith("proteomics:")]


def _select_hierarchical_comparisons(sig_df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    selected: list[str] = []
    prot_selected: list[str] = []
    tx_selected: list[str] = []

    st.subheader("Proteomics")
    prot_df = sig_df[sig_df["source_dataset"] == "proteomics"].copy()
    if prot_df.empty:
        st.caption("No proteomics comparisons available.")
    else:
        prot_files = sorted(prot_df["source_file"].dropna().astype(str).unique().tolist())
        prot_base_labels = [_friendly_proteomics_label(source_file) for source_file in prot_files]
        prot_label_counts = pd.Series(prot_base_labels).value_counts().to_dict() if prot_base_labels else {}
        prot_dataset_selected = _render_checkbox_choices(
            "Proteomics datasets",
            [
                (
                    source_file,
                    _friendly_proteomics_label(source_file)
                    if prot_label_counts.get(_friendly_proteomics_label(source_file), 0) <= 1
                    else f"{_friendly_proteomics_label(source_file)} ({Path(source_file).stem})",
                )
                for source_file in prot_files
            ],
            "prot_dataset_pick",
            default_selected=prot_files,
            columns=2,
        )

        for source_file in prot_dataset_selected:
            dataset_df = prot_df[prot_df["source_file"].astype(str) == source_file].copy()
            comparisons = sorted(dataset_df["comparison_id"].dropna().astype(str).unique().tolist())
            if not comparisons:
                continue
            dataset_title = _friendly_proteomics_label(source_file)
            if prot_label_counts.get(dataset_title, 0) > 1:
                dataset_title = f"{dataset_title} ({Path(source_file).stem})"
            with st.expander(f"Dataset: {dataset_title}", expanded=False):
                chosen = _render_checkbox_choices(
                    "Comparisons",
                    [(comparison_id, comparison_id) for comparison_id in comparisons],
                    f"prot:{Path(source_file).stem}:comparison_pick",
                    default_selected=comparisons[: min(3, len(comparisons))],
                    columns=2,
                )
                prot_selected.extend(chosen)
                selected.extend(chosen)

    st.subheader("snRNAseq")
    sn_sources = {
        "regional_snrnaseq": "Regional snRNAseq",
        "hfd_snrnaseq": "HFD snRNAseq",
    }
    sn_df = sig_df[sig_df["source_dataset"].isin(sn_sources.keys())].copy()
    sn_available = set(sn_df["source_dataset"].dropna().astype(str).unique().tolist())

    sn_selected = _render_checkbox_choices(
        "snRNAseq datasets",
        [(source_dataset, label) for source_dataset, label in sn_sources.items()],
        "sn_source_pick",
        default_selected=[source for source in sn_sources if source in sn_available],
        columns=2,
        disabled_values={source for source in sn_sources if source not in sn_available},
    )

    for source_dataset, source_label in sn_sources.items():
        source_available = source_dataset in sn_available
        with st.expander(source_label, expanded=False):
            if not source_available:
                st.caption("This dataset is currently unavailable in the consolidated table.")
                continue

            if source_dataset not in sn_selected:
                _clear_state_prefix(f"{source_dataset}:level_pick:")
                _clear_state_prefix(f"{source_dataset}:celltype_pick:")
                _clear_state_prefix(f"{source_dataset}:comparison_pick:")
                continue

            source_df = sn_df[sn_df["source_dataset"] == source_dataset].copy()
            level_options = sorted(source_df["manual_ct_lvl"].dropna().astype(str).unique().tolist())
            if not level_options:
                st.info("No levels available for this dataset.")
                continue

            level_choice = _render_checkbox_choices(
                "Levels",
                [(level, level) for level in level_options],
                f"{source_dataset}:level_pick",
                default_selected=level_options,
                columns=3,
            )
            if not level_choice:
                _clear_state_prefix(f"{source_dataset}:celltype_pick:")
                _clear_state_prefix(f"{source_dataset}:comparison_pick:")
                st.info("Select at least one level to continue.")
                continue

            level_signature_key = f"{source_dataset}:level_signature"
            level_signature = "|".join(sorted(level_choice))
            previous_signature = st.session_state.get(level_signature_key)
            if previous_signature is not None and previous_signature != level_signature:
                _clear_state_prefix(f"{source_dataset}:celltype_pick:")
                _clear_state_prefix(f"{source_dataset}:comparison_pick:")
            st.session_state[level_signature_key] = level_signature

            source_df = source_df[source_df["manual_ct_lvl"].isin(level_choice)].copy()
            celltype_options = sorted(source_df["celltype_label"].dropna().astype(str).unique().tolist())
            if not celltype_options:
                _clear_state_prefix(f"{source_dataset}:comparison_pick:")
                st.info("No celltypes match the selected levels.")
                continue

            celltype_choice = _render_checkbox_choices(
                "Celltypes",
                [(celltype, celltype) for celltype in celltype_options],
                f"{source_dataset}:celltype_pick",
                default_selected=celltype_options,
                columns=2,
            )
            if not celltype_choice:
                _clear_state_prefix(f"{source_dataset}:comparison_pick:")
                st.info("Select at least one celltype to continue.")
                continue

            source_df = source_df[source_df["celltype_label"].isin(celltype_choice)].copy()
            comparisons = sorted(source_df["comparison_id"].dropna().astype(str).unique().tolist())
            if not comparisons:
                st.info("No comparisons match the selected filters.")
                continue

            chosen = _render_checkbox_choices(
                "Comparisons",
                [(comparison_id, comparison_id) for comparison_id in comparisons],
                f"{source_dataset}:comparison_pick",
                default_selected=comparisons[: min(5, len(comparisons))],
                columns=2,
            )
            tx_selected.extend(chosen)
            selected.extend(chosen)

    st.subheader("Bulk RNAseq")
    bulk_sources = {
        "bulk_rnaseq": "Bulk RNAseq",
        "organoid_bulk_rnaseq": "Organoid bulk RNAseq",
    }
    bulk_df = sig_df[sig_df["source_dataset"].isin(bulk_sources.keys())].copy()
    bulk_available = set(bulk_df["source_dataset"].dropna().astype(str).unique().tolist())

    bulk_selected = _render_checkbox_choices(
        "Bulk datasets",
        [(source_dataset, label) for source_dataset, label in bulk_sources.items()],
        "bulk_source_pick",
        default_selected=[source for source in bulk_sources if source in bulk_available],
        columns=2,
        disabled_values={source for source in bulk_sources if source not in bulk_available},
    )

    if not bulk_available:
        st.info("Bulk RNAseq section is visible but disabled until bulk comparisons are loaded.")

    for source_dataset, source_label in bulk_sources.items():
        source_available = source_dataset in bulk_available
        with st.expander(source_label, expanded=False):
            if not source_available:
                st.caption("This dataset is currently unavailable in the consolidated table.")
                continue

            if source_dataset not in bulk_selected:
                _clear_state_prefix(f"{source_dataset}:comparison_pick:")
                continue

            source_df = bulk_df[bulk_df["source_dataset"] == source_dataset].copy()
            comparisons = sorted(source_df["comparison_id"].dropna().astype(str).unique().tolist())
            if not comparisons:
                st.info("No comparisons available for this dataset.")
                continue

            chosen = _render_checkbox_choices(
                "Comparisons",
                [(comparison_id, comparison_id) for comparison_id in comparisons],
                f"{source_dataset}:comparison_pick",
                default_selected=comparisons[: min(5, len(comparisons))],
                columns=2,
            )
            tx_selected.extend(chosen)
            selected.extend(chosen)

    return _dedupe_preserve_order(selected), _dedupe_preserve_order(prot_selected), _dedupe_preserve_order(tx_selected)


def _build_overlap_sets(
    sig_df: pd.DataFrame,
    selected_comparisons: list[str],
    prot_selected: list[str],
    tx_selected: list[str],
    include_combined_prot: bool,
    include_combined_tx: bool,
) -> dict[str, set[str]]:
    overlap_sets = gene_sets_for_comparisons(sig_df, selected_comparisons)

    if include_combined_prot and prot_selected:
        prot_union: set[str] = set()
        for comp in prot_selected:
            prot_union.update(overlap_sets.get(comp, set()))
        overlap_sets["ALL_PROTEOMICS_SELECTED"] = prot_union

    if include_combined_tx and tx_selected:
        tx_union: set[str] = set()
        for comp in tx_selected:
            tx_union.update(overlap_sets.get(comp, set()))
        overlap_sets["ALL_TRANSCRIPTOMICS_SELECTED"] = tx_union

    return overlap_sets


def _compute_intersections_from_sets(
    sig_df: pd.DataFrame,
    overlap_sets: dict[str, set[str]],
    selected_comparisons: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not overlap_sets:
        empty_summary = pd.DataFrame(columns=["intersection", "n_genes", "best_adj_pval"])
        empty_genes = pd.DataFrame(columns=["gene", "intersection", "n_comparisons", "best_adj_pval"])
        return empty_summary, empty_genes

    sig_subset = sig_df[sig_df["comparison_id"].isin(selected_comparisons)][["feature_name", "p_adj"]].copy()
    sig_subset["feature_name"] = sig_subset["feature_name"].astype(str)
    sig_subset["p_adj"] = pd.to_numeric(sig_subset["p_adj"], errors="coerce")
    best_adj_by_gene = sig_subset.groupby("feature_name")["p_adj"].min()

    memberships_by_gene: dict[str, list[str]] = defaultdict(list)
    for set_name, members in overlap_sets.items():
        for gene in members:
            memberships_by_gene[str(gene)].append(set_name)

    rows: list[dict[str, object]] = []
    for gene, membership_values in memberships_by_gene.items():
        memberships = sorted(membership_values)
        if len(memberships) < 2:
            continue
        label = " ∩ ".join(memberships)
        best_adj = best_adj_by_gene.get(gene, pd.NA)
        rows.append(
            {
                "gene": gene,
                "intersection": label,
                "n_comparisons": len(memberships),
                "best_adj_pval": best_adj,
            }
        )

    gene_table = pd.DataFrame(rows)
    if gene_table.empty:
        empty_summary = pd.DataFrame(columns=["intersection", "n_genes", "best_adj_pval"])
        empty_genes = pd.DataFrame(columns=["gene", "intersection", "n_comparisons", "best_adj_pval"])
        return empty_summary, empty_genes

    summary = (
        gene_table.groupby("intersection", as_index=False)
        .agg(
            n_genes=("gene", "count"),
            best_adj_pval=("best_adj_pval", "min"),
        )
        .sort_values(
            ["n_genes", "best_adj_pval", "intersection"],
            ascending=[False, True, True],
            na_position="last",
        )
        .reset_index(drop=True)
    )
    summary.insert(0, "intersection_id", range(1, len(summary) + 1))
    gene_table = gene_table.sort_values(["best_adj_pval", "gene"], na_position="last").reset_index(drop=True)
    return summary, gene_table


def _selected_overlap_export(
    sig_df: pd.DataFrame,
    genes: list[str],
    selected_comparisons: list[str],
) -> pd.DataFrame:
    if not genes or not selected_comparisons:
        return pd.DataFrame()

    subset = sig_df[
        sig_df["comparison_id"].isin(selected_comparisons)
        & sig_df["feature_name"].astype(str).isin(genes)
    ].copy()
    if subset.empty:
        return subset

    subset = subset.rename(
        columns={
            "feature_name": "gene",
            "source_dataset": "dataset",
            "source_file": "source_file",
            "comparison_id": "comparison",
            "log2fc_or_fc": "foldchange",
            "p_value": "pval",
            "p_adj": "adj_pval",
            "avg_value": "raw_mean_expression",
            "median_value": "raw_median_expression",
        }
    )
    subset["up_in"] = pd.NA
    foldchange = pd.to_numeric(subset["foldchange"], errors="coerce")
    subset.loc[foldchange > 0, "up_in"] = subset.loc[foldchange > 0, "group_b"]
    subset.loc[foldchange < 0, "up_in"] = subset.loc[foldchange < 0, "group_a"]
    subset["value_transform"] = [
        describe_value_transform(str(dataset), str(source_file) if pd.notna(source_file) else None)
        for dataset, source_file in zip(subset["dataset"], subset["source_file"])
    ]
    keep = [
        "gene",
        "dataset",
        "comparison",
        "up_in",
        "foldchange",
        "raw_mean_expression",
        "raw_median_expression",
        "value_transform",
        "pval",
        "adj_pval",
        "modality",
        "celltype_label",
        "manual_ct_lvl",
        "group_a",
        "group_b",
    ]
    return subset[keep].sort_values(["gene", "dataset", "comparison"]).reset_index(drop=True)


def _overlap_wide_export(long_df: pd.DataFrame) -> pd.DataFrame:
    if long_df.empty:
        return pd.DataFrame()

    pieces = []
    for value_col, prefix in [
        ("raw_mean_expression", "expr"),
        ("value_transform", "transform"),
        ("pval", "pval"),
        ("adj_pval", "padj"),
    ]:
        part = long_df.pivot_table(
            index="gene",
            columns="comparison",
            values=value_col,
            aggfunc="first",
            observed=True,
        )
        part = part.add_prefix(f"{prefix}__")
        pieces.append(part)

    out = pd.concat(pieces, axis=1).reset_index()
    return out


def _render_plotting_page(sig_df: pd.DataFrame) -> None:
    st.header("3. Plotting")
    st.caption(
        "Values are z-scored within each selected dataset before combining plots, so separately run datasets are visually comparable."
    )
    with st.expander("How z-scoring is applied"):
        st.markdown(
            """
            The plotting page uses the raw values returned by the data loaders: SCT-normalized
            snRNAseq means from the aggregated expression table and proteomics assay intensities
            from the cached expression tables. Those values are already normalized within their
            own pipelines, but they are not on the same scale across datasets.

            Z-scoring is applied only after the selected expression frames are assembled, and it is
            done independently within each `dataset_label`. That means each dataset is centered to
            mean 0 and scaled to standard deviation 1 for visualization only. The download buttons
            keep the raw, pre-z-score values.
            """
        )

    source_mode = st.radio(
        "Gene source",
        ["From overlap selection", "Manual gene input"],
        horizontal=True,
    )

    selected_genes: list[str] = []
    if source_mode == "From overlap selection":
        selected_genes = st.session_state.get("selected_intersection_genes", [])
        st.write(f"Genes from overlap selection: {len(selected_genes)}")
        if selected_genes:
            st.dataframe(pd.DataFrame({"gene": selected_genes}), width='stretch')
    else:
        text = st.text_area("Enter genes (comma or newline separated)", "LGR5\nMUC2\nEPCAM")
        selected_genes = parse_gene_text(text)

    if not selected_genes:
        st.info("Select or enter at least one gene to render expression plots.")
        return

    with st.form("plotting_form"):
        dataset_sel = _render_checkbox_choices(
            "Select datasets for expression plotting",
            _expression_dataset_options(),
            "plot_dataset_pick",
            default_selected=["regional_snrnaseq"],
            columns=2,
        )
        manual_lvl = st.selectbox("snRNAseq level", ["lvl0", "lvl1", "lvl2", "lvl3", "lvl4"], index=1)
        col_left, col_right = st.columns(2)
        split_by_segment = col_left.checkbox(
            "Split Regional snRNAseq by segment",
            value=False,
            help="Show one column per celltype x segment (Duodenum/Jejunum/Ileum/Colon) "
            "instead of segment-pooled means.",
            disabled=not REGIONAL_SEG_PARQUET.exists(),
        )
        grouping_mode = col_right.radio(
            "Column grouping (snRNAseq)",
            ["Celltype first", "Condition first (diet / segment)"],
            help="'Condition first' clusters columns by diet (HFD) or segment (Regional) "
            "so conditions sit next to each other.",
        )
        condition_first = grouping_mode.startswith("Condition")
        layout_mode = st.radio("Layout", ["Combined", "Separate panels"], horizontal=True)
        plot_type = st.radio("Plot type", ["Heatmap", "Dotplot"], horizontal=True)
        plotting_submitted = st.form_submit_button(
            "Start plotting",
            type="primary",
            disabled=not dataset_sel,
        )

    plotting_signature = _plot_selection_signature(
        source_mode,
        selected_genes,
        dataset_sel,
        manual_lvl,
        layout_mode,
        plot_type,
        split_by_segment,
        condition_first,
    )

    if plotting_submitted:
        if not dataset_sel:
            st.info("Select at least one dataset.")
            return

        expr_frames: list[pd.DataFrame] = []
        if "regional_snrnaseq" in dataset_sel:
            if split_by_segment and REGIONAL_SEG_PARQUET.exists():
                regional_expr = load_regional_segment_expression(
                    REGIONAL_SEG_PARQUET,
                    genes=selected_genes,
                    manual_ct_lvl=manual_lvl,
                    condition_first=condition_first,
                )
                if not regional_expr.empty:
                    expr_frames.append(regional_expr)
            else:
                expr_frames.append(
                    load_snrna_expression(
                        SNRNA_AGG_CSV,
                        genes=selected_genes,
                        manual_ct_lvl=manual_lvl,
                        dataset_label=f"regional_snrnaseq_{manual_lvl}",
                    )
                )

        hfd_tissues = [token.split(":", 1)[1] for token in dataset_sel if token.startswith("hfd_snrnaseq:")]
        if hfd_tissues:
            hfd_expr = load_hfd_expression(
                HFD_AGG_PARQUET,
                genes=selected_genes,
                manual_ct_lvl=manual_lvl,
                tissues=hfd_tissues,
                condition_first=condition_first,
            )
            if not hfd_expr.empty:
                expr_frames.append(hfd_expr)

        prot_selected = _parse_proteomics_selection(dataset_sel)
        if prot_selected:
            prot_expr_all = _load_proteomics_expr(str(PROTEOMICS_DIR), tuple(sorted(set(selected_genes))))
            if not prot_expr_all.empty:
                used_labels: set[str] = set()
                for stem in prot_selected:
                    internal_label = f"proteomics:{stem}"
                    match = prot_expr_all[prot_expr_all["dataset_label"] == internal_label]
                    if match.empty:
                        continue
                    match = match.copy()
                    friendly = _friendly_proteomics_label_from_stem(stem)
                    if friendly in used_labels:
                        friendly = f"{friendly} ({stem})"
                    used_labels.add(friendly)
                    match["dataset_label"] = friendly
                    expr_frames.append(match)

        if not expr_frames:
            st.warning("No expression data found for selected genes and datasets.")
            return

        expr_df = pd.concat(expr_frames, ignore_index=True)
        expr_df = zscore_within_dataset(expr_df, value_col="value")
        st.session_state["plotting_results"] = {
            "signature": plotting_signature,
            "expr_df": expr_df,
        }

    if not dataset_sel:
        st.info("Select at least one dataset and click Start plotting.")
        return

    cached = st.session_state.get("plotting_results")
    if not cached or cached.get("signature") != plotting_signature:
        st.info("Finalize selections above, then click Start plotting.")
        return

    expr_df = cached["expr_df"]

    if layout_mode == "Combined" and plot_type == "Heatmap":
        fig = heatmap_combined(expr_df, value_col="z_value", title="Combined z-scored expression heatmap")
    elif layout_mode == "Combined" and plot_type == "Dotplot":
        fig = dotplot_combined(expr_df, value_col="z_value", title="Combined z-scored expression dotplot")
    elif layout_mode == "Separate panels" and plot_type == "Heatmap":
        fig = heatmap_faceted(expr_df, value_col="z_value", title="Faceted z-scored expression heatmap")
    else:
        fig = dotplot_faceted(expr_df, value_col="z_value", title="Faceted z-scored expression dotplot")

    st.pyplot(fig, clear_figure=True)

    with st.expander("Expression data used for plotting"):
        st.dataframe(expr_df.sort_values(["dataset_label", "gene", "group_label"]).reset_index(drop=True), width='stretch')


def _render_overlap_page(sig_df: pd.DataFrame) -> None:
    st.header("2. Compare Significant Hits Across Analyses")
    st.caption("Use the hierarchical selectors below to narrow from modality to dataset and comparison.")
    with st.form("overlap_analysis_form"):
        selected_comparisons, prot_selected, tx_selected = _select_hierarchical_comparisons(sig_df)

        col_a, col_b = st.columns(2)
        include_combined_prot = col_a.checkbox(
            "Add combined set: ALL selected proteomics",
            value=True,
        )
        include_combined_tx = col_b.checkbox(
            "Add combined set: ALL selected transcriptomics",
            value=True,
        )

        scale_mode = st.radio(
            "Intersection size scale",
            ["Linear", "Log2"],
            horizontal=True,
            key="overlap_scale",
        )

        submitted = st.form_submit_button(
            "Start overlap analysis",
            type="primary",
            disabled=len(selected_comparisons) < 2,
        )

    current_signature = _selection_signature(
        selected_comparisons,
        prot_selected,
        tx_selected,
        include_combined_prot,
        include_combined_tx,
    )
    if submitted:
        if len(selected_comparisons) < 2:
            st.info("Select at least two comparisons to compute overlaps.")
            return

        overlap_sets = _build_overlap_sets(
            sig_df,
            selected_comparisons,
            prot_selected,
            tx_selected,
            include_combined_prot,
            include_combined_tx,
        )
        summary, gene_intersections = _compute_intersections_from_sets(sig_df, overlap_sets, selected_comparisons)
        st.session_state["overlap_results"] = {
            "signature": current_signature,
            "overlap_sets": overlap_sets,
            "summary": summary,
            "gene_intersections": gene_intersections,
        }

    cached = st.session_state.get("overlap_results")
    if len(selected_comparisons) < 2:
        st.info(f"Selected comparisons: {len(selected_comparisons)}. Select at least two to compute overlaps.")
        return

    if not cached or cached.get("signature") != current_signature:
        st.info("Finalize the selections above, then click Start overlap analysis.")
        return

    overlap_sets = cached["overlap_sets"]
    summary = cached["summary"]
    gene_intersections = cached["gene_intersections"]

    fig = plot_overlap(
        overlap_sets,
        title="Significant hit overlap",
        log_scale=(scale_mode == "Log2"),
        intersection_order=summary["intersection"].tolist() if not summary.empty else None,
    )
    st.pyplot(fig, clear_figure=True)

    st.subheader("Intersections")
    st.dataframe(summary.set_index("intersection_id"), width='stretch')

    if summary.empty:
        st.warning("No intersections found for the selected comparisons.")
        return

    intersection_meta = {
        row["intersection"]: (int(row["intersection_id"]), int(row["n_genes"]))
        for _, row in summary.iterrows()
    }

    def _format_intersection(label: str) -> str:
        intersection_id, n_genes = intersection_meta.get(label, ("?", "?"))
        return f"#{intersection_id} · {n_genes} genes · {label}"

    selected_intersection = st.selectbox(
        "Choose intersection to inspect genes (IDs match the plot)",
        options=summary["intersection"].tolist(),
        format_func=_format_intersection,
    )
    genes_table = gene_intersections[gene_intersections["intersection"] == selected_intersection].copy()
    genes_table = genes_table[["gene", "n_comparisons", "best_adj_pval"]].reset_index(drop=True)
    st.dataframe(genes_table, width='stretch')

    selected_gene_export = _selected_overlap_export(sig_df, genes_table["gene"].tolist(), selected_comparisons)
    if not selected_gene_export.empty:
        export_cols = [
            "gene",
            "dataset",
            "comparison",
            "up_in",
            "foldchange",
            "raw_mean_expression",
            "raw_median_expression",
            "value_transform",
            "pval",
            "adj_pval",
            "modality",
            "celltype_label",
            "manual_ct_lvl",
            "group_a",
            "group_b",
        ]
        st.download_button(
            "Download selected gene set (CSV)",
            data=selected_gene_export[export_cols].to_csv(index=False).encode("utf-8"),
            file_name="selected_overlap_gene_set.csv",
            mime="text/csv",
        )

        wide_export = _overlap_wide_export(selected_gene_export[export_cols])
        st.download_button(
            "Download selected gene set (wide raw matrix CSV)",
            data=wide_export.to_csv(index=False).encode("utf-8"),
            file_name="selected_overlap_gene_set_wide.csv",
            mime="text/csv",
        )

    if st.button("Use these genes in Plotting page"):
        st.session_state["selected_intersection_genes"] = genes_table["gene"].tolist()
        st.success("Gene set saved. Open page '3. Plotting'.")


def _render_about_page() -> None:
    st.markdown(
        """
        This app explores integrated significant-hit tables from several pig gut multi-omics
        datasets side by side: regional and high-fat-diet (HFD) snRNAseq pseudobulk differential
        expression, and ECM / soluble / villus proteomics comparisons. All results are normalized
        into one common long-format table (gene, comparison, direction, fold change, p-value,
        modality, cell type) so they can be queried and compared together, without needing to
        touch the original per-pipeline outputs.
        """
    )
    st.markdown("**What you can do, page by page:**")
    st.markdown(
        """
        - **1. Gene Query** — look up one or more genes and see every significant hit for them
          across all datasets and comparisons, with a CSV download.
        - **2. Compare Significant Hits** — pick comparisons across proteomics and
          transcriptomics datasets and see how their significant gene sets overlap, as an
          UpSet-style intersection plot.
        - **3. Plotting** — visualize expression of a gene set (from the overlap selection or
          typed manually) as dotplots or heatmaps across datasets and cell types, z-scored per
          dataset so differently-scaled analyses stay visually comparable.
        """
    )
    st.caption(
        "Data provenance: pseudobulk edgeR differential expression on CellBender-corrected "
        "snRNAseq counts, and log2 intensity proteomics from SummarizedExperiment objects, "
        "consolidated into one significant-findings table. See methods_webapp_paragraph.md in "
        "the repository for full methodological detail."
    )


def _render_query_page(sig_df: pd.DataFrame) -> None:
    st.header("1. Gene / Gene-List Query")
    text = st.text_area("Enter one or more genes (comma or newline separated)", "LGR5")
    genes = parse_gene_text(text)

    if not genes:
        st.info("Enter at least one gene.")
        return

    results = query_genes(sig_df, genes)
    if results.empty:
        st.warning("No significant detections found for the provided genes.")
        return

    display_cols = [
        "gene",
        "dataset",
        "comparison",
        "up_in",
        "foldchange",
        "raw_mean_expression",
        "raw_median_expression",
        "value_transform",
        "pval",
        "adj_pval",
        "modality",
        "celltype_label",
        "manual_ct_lvl",
        "group_a",
        "group_b",
    ]
    st.dataframe(results[display_cols], width='stretch')

    csv_data = results[display_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download query results (CSV)",
        data=csv_data,
        file_name="gene_query_results.csv",
        mime="text/csv",
    )


def main() -> None:
    st.title("Multi-Omics Explorer")

    if not PARQUET_PATH.exists():
        st.error(
            f"Missing consolidated dataset: {PARQUET_PATH}. "
            "Run build_overview_dataset.py first."
        )
        return

    page = st.sidebar.radio(
        "Navigate",
        [
            "About",
            "1. Gene Query",
            "2. Compare Significant Hits",
            "3. Plotting",
        ],
    )

    if page == "About":
        _render_about_page()
        return

    sig_df = _load_significance(str(PARQUET_PATH))

    st.sidebar.caption(f"Rows: {len(sig_df):,}")
    st.sidebar.caption(f"Comparisons: {sig_df['comparison_id'].nunique():,}")

    if page == "1. Gene Query":
        _render_query_page(sig_df)
    elif page == "2. Compare Significant Hits":
        _render_overlap_page(sig_df)
    else:
        _render_plotting_page(sig_df)


if __name__ == "__main__":
    main()
