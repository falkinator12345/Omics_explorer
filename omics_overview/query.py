from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


def _load_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError("Query input must be .parquet or .csv")


def find_feature(
    feature: str,
    dataset_path: str | Path = "consolidated_significant_findings.parquet",
    source_dataset: Optional[str] = None,
    modality: Optional[str] = None,
) -> pd.DataFrame:
    df = _load_table(dataset_path)
    out = df[df["feature_name"].astype(str).str.lower() == feature.lower()]
    if source_dataset:
        out = out[out["source_dataset"] == source_dataset]
    if modality:
        out = out[out["modality"] == modality]
    return out.sort_values(["source_dataset", "comparison_id"]).reset_index(drop=True)


def list_comparisons(
    dataset_path: str | Path = "consolidated_significant_findings.parquet",
    source_dataset: Optional[str] = None,
    modality: Optional[str] = None,
) -> pd.DataFrame:
    df = _load_table(dataset_path)
    if source_dataset:
        df = df[df["source_dataset"] == source_dataset]
    if modality:
        df = df[df["modality"] == modality]

    cols = [
        "comparison_id",
        "source_dataset",
        "modality",
        "sample_class",
        "group_a",
        "group_b",
        "celltype_label",
        "manual_ct_lvl",
    ]
    return df[cols].drop_duplicates().sort_values(cols).reset_index(drop=True)


def filter_significant(
    dataset_path: str | Path = "consolidated_significant_findings.parquet",
    min_abs_fc: Optional[float] = None,
    max_p_adj: Optional[float] = None,
    direction: Optional[str] = None,
) -> pd.DataFrame:
    df = _load_table(dataset_path)

    if min_abs_fc is not None:
        df = df[df["log2fc_or_fc"].abs() >= float(min_abs_fc)]
    if max_p_adj is not None:
        df = df[df["p_adj"] <= float(max_p_adj)]
    if direction:
        df = df[df["direction"].astype(str).str.lower() == direction.lower()]

    return df.reset_index(drop=True)


def overlap_features(
    comparison_ids: Iterable[str],
    dataset_path: str | Path = "consolidated_significant_findings.parquet",
) -> dict[str, set[str]]:
    df = _load_table(dataset_path)
    comparison_ids = list(comparison_ids)
    out: dict[str, set[str]] = {}

    for comp in comparison_ids:
        features = set(
            df.loc[df["comparison_id"] == comp, "feature_name"].dropna().astype(str).tolist()
        )
        out[comp] = features
    return out
