from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pandas as pd

from omics_overview.parsers import infer_sample_class_from_name
from omics_overview.visualize import load_aggregated_expression


def load_significance_table(parquet_path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(parquet_path)


def parse_gene_text(text: str) -> list[str]:
    raw = text.replace("\n", ",").split(",")
    genes = [token.strip() for token in raw if token.strip()]
    # Keep user order while removing duplicates.
    seen = set()
    out = []
    for gene in genes:
        key = gene.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(gene)
    return out


def describe_value_transform(source_dataset: str, source_file: str | None = None) -> str:
    if source_dataset in {"regional_snrnaseq", "hfd_snrnaseq"}:
        return "SCT-normalized, log-scaled aggregated mean"
    if source_dataset in {"bulk_rnaseq", "organoid_bulk_rnaseq"}:
        return "normalized bulk RNAseq aggregated mean"
    if source_dataset == "proteomics":
        sample_class = infer_sample_class_from_name(source_file or "") if source_file else "Unknown"
        if sample_class and sample_class != "Unknown":
            return f"log2 intensity, condition-averaged ({sample_class})"
        return "log2 intensity, condition-averaged"
    return "raw expression"


def query_genes(df: pd.DataFrame, genes: Iterable[str]) -> pd.DataFrame:
    genes = list(genes)
    if not genes:
        return pd.DataFrame()

    wanted = {gene.casefold() for gene in genes}
    hits = df[df["feature_name"].astype(str).str.casefold().isin(wanted)].copy()
    if hits.empty:
        return hits

    out = hits[
        [
            "source_dataset",
            "source_file",
            "comparison_id",
            "direction",
            "log2fc_or_fc",
            "p_value",
            "p_adj",
            "avg_value",
            "median_value",
            "modality",
            "celltype_label",
            "manual_ct_lvl",
            "group_a",
            "group_b",
            "feature_name",
        ]
    ].rename(
        columns={
            "source_dataset": "dataset",
            "source_file": "source_file",
            "comparison_id": "comparison",
            "log2fc_or_fc": "foldchange",
            "p_value": "pval",
            "p_adj": "adj_pval",
            "avg_value": "raw_mean_expression",
            "median_value": "raw_median_expression",
            "feature_name": "gene",
        }
    )
    out["up_in"] = pd.NA
    foldchange = pd.to_numeric(out["foldchange"], errors="coerce")
    out.loc[foldchange > 0, "up_in"] = out.loc[foldchange > 0, "group_b"]
    out.loc[foldchange < 0, "up_in"] = out.loc[foldchange < 0, "group_a"]
    out["value_transform"] = [
        describe_value_transform(str(dataset), str(source_file) if pd.notna(source_file) else None)
        for dataset, source_file in zip(out["dataset"], out["source_file"])
    ]
    return out.sort_values(["adj_pval", "pval"], na_position="last").reset_index(drop=True)


def gene_sets_for_comparisons(df: pd.DataFrame, comparison_ids: list[str]) -> dict[str, set[str]]:
    if not comparison_ids:
        return {}

    filtered = df.loc[
        df["comparison_id"].isin(comparison_ids),
        ["comparison_id", "feature_name"],
    ].dropna(subset=["comparison_id", "feature_name"])
    if filtered.empty:
        return {comp: set() for comp in comparison_ids}

    grouped = filtered.groupby("comparison_id")["feature_name"].agg(
        lambda values: set(values.astype(str).tolist())
    )
    return {comp: grouped.get(comp, set()) for comp in comparison_ids}


def intersection_summary(df: pd.DataFrame, comparison_ids: list[str]) -> pd.DataFrame:
    if len(comparison_ids) < 2:
        return pd.DataFrame(columns=["intersection", "n_genes", "best_adj_pval"])

    subset = df[df["comparison_id"].isin(comparison_ids)].copy()
    if subset.empty:
        return pd.DataFrame(columns=["intersection", "n_genes", "best_adj_pval"])

    grouped = subset.groupby("feature_name")
    rows = []
    for gene, block in grouped:
        comps = sorted(block["comparison_id"].astype(str).unique().tolist())
        label = " ∩ ".join(comps)
        best_adj = pd.to_numeric(block["p_adj"], errors="coerce").min()
        rows.append({
            "gene": gene,
            "intersection": label,
            "n_comparisons": len(comps),
            "best_adj_pval": best_adj,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["intersection", "n_genes", "best_adj_pval"])

    summary = (
        out.groupby("intersection", as_index=False)
        .agg(
            n_genes=("gene", "count"),
            best_adj_pval=("best_adj_pval", "min"),
        )
        .sort_values(["n_genes", "best_adj_pval"], ascending=[False, True], na_position="last")
        .reset_index(drop=True)
    )
    return summary


def genes_in_intersection(df: pd.DataFrame, intersection_label: str) -> pd.DataFrame:
    if not intersection_label:
        return pd.DataFrame(columns=["gene", "n_comparisons", "best_adj_pval"])

    selected = [token.strip() for token in intersection_label.split("∩") if token.strip()]
    subset = df[df["comparison_id"].isin(selected)].copy()
    if subset.empty:
        return pd.DataFrame(columns=["gene", "n_comparisons", "best_adj_pval"])

    grouped = subset.groupby("feature_name")
    rows = []
    for gene, block in grouped:
        comps = sorted(block["comparison_id"].astype(str).unique().tolist())
        label = " ∩ ".join(comps)
        if label != intersection_label:
            continue
        rows.append(
            {
                "gene": gene,
                "n_comparisons": len(comps),
                "best_adj_pval": pd.to_numeric(block["p_adj"], errors="coerce").min(),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["gene", "n_comparisons", "best_adj_pval"])
    return out.sort_values(["best_adj_pval", "gene"], na_position="last").reset_index(drop=True)


def load_snrna_expression(
    csv_path: str | Path,
    genes: list[str],
    manual_ct_lvl: str,
    dataset_label: str | None = None,
) -> pd.DataFrame:
    expr = load_aggregated_expression(csv_path, manual_ct_lvl=manual_ct_lvl, genes=genes)
    if expr.empty:
        return pd.DataFrame(
            columns=[
                "gene",
                "dataset_label",
                "group_label",
                "value",
                "value_transform",
                "modality",
                "source_dataset",
                "sample_class",
                "celltype_label",
                "manual_ct_lvl",
            ]
        )

    label = dataset_label or f"regional_snrnaseq_{manual_ct_lvl}"
    out = expr.rename(columns={"celltype_label": "group_label", "mean": "value"}).copy()
    out["dataset_label"] = label
    out["modality"] = "transcriptomics"
    out["source_dataset"] = "regional_snrnaseq"
    out["sample_class"] = manual_ct_lvl
    out["celltype_label"] = out["group_label"]
    out["manual_ct_lvl"] = manual_ct_lvl
    out["value_transform"] = describe_value_transform("regional_snrnaseq")
    return out[
        [
            "gene",
            "dataset_label",
            "group_label",
            "value",
            "value_transform",
            "modality",
            "source_dataset",
            "sample_class",
            "celltype_label",
            "manual_ct_lvl",
        ]
    ]


EXPRESSION_COLUMNS = [
    "gene",
    "dataset_label",
    "group_label",
    "value",
    "value_transform",
    "modality",
    "source_dataset",
    "sample_class",
    "celltype_label",
    "manual_ct_lvl",
]


def load_hfd_expression(
    parquet_path: str | Path,
    genes: list[str],
    manual_ct_lvl: str,
    tissues: Iterable[str] = ("Colon", "Jejunum"),
    condition_first: bool = False,
) -> pd.DataFrame:
    """Load HFD snRNAseq aggregated expression (per celltype x diet) from parquet.

    Mirrors the regional loader but keeps diet in the group label so CD/HFDL/HFDO
    can be compared side by side. With ``condition_first=True`` the group labels
    are "diet | celltype" so columns cluster by diet instead of celltype.
    """
    parquet_path = Path(parquet_path)
    if not parquet_path.exists():
        return pd.DataFrame(columns=EXPRESSION_COLUMNS)

    level_col = manual_ct_lvl if manual_ct_lvl.startswith("Manual_CT_") else f"Manual_CT_{manual_ct_lvl}"
    lvl_token = level_col.replace("Manual_CT_", "")

    expr = pd.read_parquet(
        parquet_path,
        columns=["tissue", "diet", level_col, "gene", "mean"],
    )
    if level_col not in expr.columns:
        return pd.DataFrame(columns=EXPRESSION_COLUMNS)

    expr = expr[expr["tissue"].isin(list(tissues))]
    if genes:
        wanted = {gene.casefold() for gene in genes}
        expr = expr[expr["gene"].astype(str).str.casefold().isin(wanted)].copy()
    if expr.empty:
        return pd.DataFrame(columns=EXPRESSION_COLUMNS)

    expr = expr.rename(columns={level_col: "celltype_label", "mean": "value"})
    expr = expr.dropna(subset=["celltype_label"])
    expr = expr[expr["celltype_label"].astype(str).str.lower() != "nan"]

    celltype = expr["celltype_label"].astype(str)
    diet = expr["diet"].astype(str)
    expr["group_label"] = (diet + " | " + celltype) if condition_first else (celltype + " | " + diet)
    expr["dataset_label"] = "hfd_snrnaseq_" + expr["tissue"].astype(str) + "_" + lvl_token
    expr["modality"] = "transcriptomics"
    expr["source_dataset"] = "hfd_snrnaseq"
    expr["sample_class"] = expr["tissue"].astype(str)
    expr["manual_ct_lvl"] = lvl_token
    expr["value_transform"] = describe_value_transform("hfd_snrnaseq")
    return expr[EXPRESSION_COLUMNS]


def load_regional_segment_expression(
    parquet_path: str | Path,
    genes: list[str],
    manual_ct_lvl: str,
    condition_first: bool = False,
) -> pd.DataFrame:
    """Load regional snRNAseq aggregated expression split by intestinal segment.

    Returns one value per celltype x segment (Duodenum/Jejunum/Ileum/Colon)
    instead of the segment-pooled means in the legacy aggregated CSV.
    """
    parquet_path = Path(parquet_path)
    if not parquet_path.exists():
        return pd.DataFrame(columns=EXPRESSION_COLUMNS)

    level_col = manual_ct_lvl if manual_ct_lvl.startswith("Manual_CT_") else f"Manual_CT_{manual_ct_lvl}"
    lvl_token = level_col.replace("Manual_CT_", "")

    expr = pd.read_parquet(
        parquet_path,
        columns=["segment", level_col, "gene", "mean"],
    )
    if genes:
        wanted = {gene.casefold() for gene in genes}
        expr = expr[expr["gene"].astype(str).str.casefold().isin(wanted)].copy()
    if expr.empty:
        return pd.DataFrame(columns=EXPRESSION_COLUMNS)

    expr = expr.rename(columns={level_col: "celltype_label", "mean": "value"})
    expr = expr.dropna(subset=["celltype_label"])
    expr = expr[expr["celltype_label"].astype(str).str.lower() != "nan"]

    celltype = expr["celltype_label"].astype(str)
    segment = expr["segment"].astype(str)
    expr["group_label"] = (segment + " | " + celltype) if condition_first else (celltype + " | " + segment)
    expr["dataset_label"] = f"regional_snrnaseq_{lvl_token}_by_segment"
    expr["modality"] = "transcriptomics"
    expr["source_dataset"] = "regional_snrnaseq"
    expr["sample_class"] = segment
    expr["manual_ct_lvl"] = lvl_token
    expr["value_transform"] = describe_value_transform("regional_snrnaseq")
    return expr[EXPRESSION_COLUMNS]


def _read_proteomics_cache(cache_path: Path) -> pd.DataFrame | None:
    try:
        cached = pd.read_csv(cache_path)
        required = {"gene", "group_label", "value"}
        if required.issubset(set(cached.columns)):
            return cached
    except Exception:
        pass
    return None


def load_proteomics_expression(proteomics_dir: str | Path, genes: list[str]) -> pd.DataFrame:
    # Reads the checked-in `*_expression_cache.csv` files directly - the .rds
    # sources stay local-only and are regenerated manually (see
    # proteomics_cache_and_phospho_operations.ipynb), so the app never shells
    # out to R at runtime.
    proteomics_dir = Path(proteomics_dir)
    frames: list[pd.DataFrame] = []

    for cache_path in sorted(proteomics_dir.glob("*_expression_cache.csv")):
        stem = cache_path.stem
        if stem.endswith("_expression_cache"):
            stem = stem[: -len("_expression_cache")]
        sample_class = infer_sample_class_from_name(stem)
        expr = _read_proteomics_cache(cache_path)
        if expr is None or expr.empty:
            continue

        if genes:
            wanted = {gene.casefold() for gene in genes}
            expr = expr[expr["gene"].astype(str).str.casefold().isin(wanted)].copy()

        expr["dataset_label"] = f"proteomics:{stem}"
        expr["sample_class"] = sample_class
        expr["modality"] = "proteomics"
        expr["source_dataset"] = "proteomics"
        expr["celltype_label"] = "bulk"
        expr["manual_ct_lvl"] = pd.NA
        expr["value_transform"] = describe_value_transform("proteomics", stem)
        frames.append(
            expr[
                [
                    "gene",
                    "dataset_label",
                    "group_label",
                    "value",
                    "value_transform",
                    "modality",
                    "source_dataset",
                    "sample_class",
                    "celltype_label",
                    "manual_ct_lvl",
                ]
            ]
        )

    if not frames:
        return pd.DataFrame(
            columns=[
                "gene",
                "dataset_label",
                "group_label",
                "value",
                "value_transform",
                "modality",
                "source_dataset",
                "sample_class",
                "celltype_label",
                "manual_ct_lvl",
            ]
        )
    return pd.concat(frames, ignore_index=True)


def zscore_within_dataset(expr_df: pd.DataFrame, value_col: str = "value") -> pd.DataFrame:
    if expr_df.empty:
        return expr_df.copy()

    out = expr_df.copy()
    out[value_col] = pd.to_numeric(out[value_col], errors="coerce")

    def _z(series: pd.Series) -> pd.Series:
        std = series.std(ddof=0)
        if pd.isna(std) or std == 0:
            return pd.Series([0.0] * len(series), index=series.index)
        return (series - series.mean()) / std

    out["z_value"] = out.groupby("dataset_label")[value_col].transform(_z)
    return out
