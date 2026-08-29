from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from .parsers import (
    enforce_schema_rules,
    load_excel_all_sheets,
    normalize_table,
    parse_bulk_rnaseq_file,
    parse_phospho_workbook_long,
    parse_proteomics_file,
    parse_proteomics_rds_long,
    parse_transcriptomics_filename,
)


def _parse_transcriptomics_folder(folder: Path, source_dataset: str) -> List[pd.DataFrame]:
    frames: List[pd.DataFrame] = []
    for path in sorted(folder.glob("*.xlsx")):
        meta = parse_transcriptomics_filename(path, source_dataset=source_dataset)
        for table in load_excel_all_sheets(path):
            normalized = normalize_table(table, meta)
            frames.append(normalized)
    return frames


def _parse_proteomics_folder(folder: Path) -> List[pd.DataFrame]:
    frames: List[pd.DataFrame] = []
    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.name.startswith("."):
            continue
        suffix = path.suffix.lower()
        if suffix not in {".xlsx", ".xls", ".rds"}:
            continue

        lower_name = path.name.lower()
        if suffix in {".xlsx", ".xls"} and ("faf107" in lower_name or "faf108" in lower_name):
            phospho = parse_phospho_workbook_long(path)
            if not phospho.empty:
                frames.append(phospho)
            continue

        meta = parse_proteomics_file(path)
        tables: List[pd.DataFrame] = []
        if suffix in {".xlsx", ".xls"}:
            tables = load_excel_all_sheets(path)
        elif suffix == ".rds":
            frames.append(parse_proteomics_rds_long(path))
            continue

        for table in tables:
            normalized = normalize_table(table, meta)
            frames.append(normalized)

    return frames


def _parse_bulk_folder(folder: Path) -> List[pd.DataFrame]:
    frames: List[pd.DataFrame] = []
    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.name.startswith("."):
            continue
        suffix = path.suffix.lower()
        if suffix not in {".xlsx", ".xls", ".csv", ".tsv"}:
            continue

        meta = parse_bulk_rnaseq_file(path)
        if suffix in {".xlsx", ".xls"}:
            tables = load_excel_all_sheets(path)
        elif suffix == ".csv":
            tables = [pd.read_csv(path)]
        else:
            tables = [pd.read_csv(path, sep="\t")]

        for table in tables:
            normalized = normalize_table(table, meta)
            frames.append(normalized)

    return frames


def build_consolidated_dataset(
    data_dir: str | Path = "Data",
    output_path: str | Path = "consolidated_significant_findings.parquet",
) -> pd.DataFrame:
    data_dir = Path(data_dir).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()

    all_frames: List[pd.DataFrame] = []

    regional_dir = data_dir / "Regional_snRNAseq_DGE_excel"
    if regional_dir.exists():
        all_frames.extend(
            _parse_transcriptomics_folder(regional_dir, source_dataset="regional_snrnaseq")
        )

    hfd_dir = data_dir / "HFD_snRNAseq_DGE_excel"
    if hfd_dir.exists():
        all_frames.extend(_parse_transcriptomics_folder(hfd_dir, source_dataset="hfd_snrnaseq"))

    proteomics_dir = data_dir / "Proteomics"
    if proteomics_dir.exists():
        all_frames.extend(_parse_proteomics_folder(proteomics_dir))

    bulk_dir = data_dir / "organoid_bulk_RNAseq"
    if bulk_dir.exists():
        all_frames.extend(_parse_bulk_folder(bulk_dir))

    if not all_frames:
        raise ValueError("No parseable input tables found in Data/. Check folder contents.")

    combined = pd.concat(all_frames, ignore_index=True)
    combined = enforce_schema_rules(combined)

    # Drop exact duplicates introduced by repeated sheets/files.
    combined = combined.drop_duplicates()

    # Sort to make output deterministic.
    combined = combined.sort_values(
        by=["source_dataset", "comparison_id", "feature_name"], kind="stable"
    ).reset_index(drop=True)

    try:
        combined.to_parquet(output_path, index=False)
    except Exception as exc:
        raise RuntimeError(
            "Failed to write parquet. Install `pyarrow` or `fastparquet`, or provide a .csv output path."
        ) from exc

    return combined
