from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional

import pandas as pd

from .schema import COLUMN_ALIASES, REQUIRED_COLUMNS, VALID_MODALITIES

TRANSCRIPTOMIC_FILENAME_RE = re.compile(
    r"^(?P<date>\d{8})_Manual_CT_(?P<manual_ct_lvl>lvl\d+)_(?P<group_a>[^_]+)_(?P<group_b>[^_]+)_(?P<celltype_label>.+?)(?:_(?P<suffix>[^.]+))?\.xlsx$",
    re.IGNORECASE,
)

HFD_FILENAME_RE = re.compile(
    r"^(?P<date>\d{8})_Manual_CT_(?P<manual_ct_lvl>lvl\d+)_(?P<segment>[^_]+)_(?P<group_a>[^_]+)_(?P<group_b>[^_]+)_(?P<celltype_label>.+?)(?:_(?P<suffix>[^.]+))?\.xlsx$",
    re.IGNORECASE,
)

_R_SCRIPT_CACHE: Optional[str] = None


@dataclass
class ParsedFileMetadata:
    source_dataset: str
    modality: str
    input_kind: str
    sample_class: str
    source_file: str
    comparison_id: str
    group_a: str
    group_b: str
    celltype_label: str
    manual_ct_lvl: Optional[str]


def normalize_colname(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def first_matching_column(df: pd.DataFrame, aliases: List[str]) -> Optional[str]:
    normalized = {normalize_colname(c): c for c in df.columns}
    for alias in aliases:
        key = normalize_colname(alias)
        if key in normalized:
            return normalized[key]
    return None


def parse_transcriptomics_filename(path: Path, source_dataset: str) -> ParsedFileMetadata:
    if source_dataset == "hfd_snrnaseq":
        match = HFD_FILENAME_RE.match(path.name)
    else:
        match = TRANSCRIPTOMIC_FILENAME_RE.match(path.name)
    if not match:
        raise ValueError(f"Unable to parse transcriptomics filename: {path.name}")

    parts = match.groupdict()
    group_a = parts["group_a"]
    group_b = parts["group_b"]
    celltype_label = parts["celltype_label"]
    manual_ct_lvl = parts["manual_ct_lvl"]
    if source_dataset == "hfd_snrnaseq":
        segment = parts.get("segment", "NA")
        comparison_id = f"{source_dataset}:{segment}:{group_a}_vs_{group_b}:{celltype_label}:{manual_ct_lvl}"
    else:
        comparison_id = f"{source_dataset}:{group_a}_vs_{group_b}:{celltype_label}:{manual_ct_lvl}"

    return ParsedFileMetadata(
        source_dataset=source_dataset,
        modality="transcriptomics",
        input_kind="dge_excel",
        sample_class="segmental" if source_dataset == "regional_snrnaseq" else "dietal",
        source_file=str(path),
        comparison_id=comparison_id,
        group_a=group_a,
        group_b=group_b,
        celltype_label=celltype_label,
        manual_ct_lvl=manual_ct_lvl,
    )


def infer_sample_class_from_name(filename: str) -> str:
    name = filename.lower()
    if "faf107" in name:
        return "Phospho Colon (FAF107)"
    if "faf108" in name:
        return "Phospho Jejunum (FAF108)"
    if "ecm" in name and "jejunum" in name:
        return "ECM Jejunum"
    if "ecm" in name and "colon" in name:
        return "ECM Colon"
    if "soluble" in name and "jejunum" in name:
        return "Soluble Jejunum"
    if "soluble" in name and "colon" in name:
        return "Soluble Colon"
    if "villus" in name and "jejunum" in name:
        return "Villus Jejunum"
    if "phospho" in name:
        return "Phospho"
    return "Unknown"


def parse_proteomics_file(path: Path) -> ParsedFileMetadata:
    sample_class = infer_sample_class_from_name(path.name)
    comparison_id = f"proteomics:{path.stem}"
    return ParsedFileMetadata(
        source_dataset="proteomics",
        modality="proteomics",
        input_kind="rds" if path.suffix.lower() == ".rds" else "excel",
        sample_class=sample_class,
        source_file=str(path),
        comparison_id=comparison_id,
        group_a="NA",
        group_b="NA",
        celltype_label="bulk",
        manual_ct_lvl=None,
    )


def _split_gene_tokens(value: object) -> list[str]:
    if pd.isna(value):
        return []
    parts = re.split(r"[;,|/]", str(value))
    return [token.strip() for token in parts if token and token.strip()]


def _select_phospho_quantity_columns(columns: List[str], mode: str) -> List[str]:
    out: List[str] = []
    for column in columns:
        key = normalize_colname(column)
        if mode == "ptm" and "ptm_quantity" in key:
            out.append(column)
        if mode == "log" and "log_quantity" in key:
            out.append(column)
    return out


def _select_phospho_pvalue_columns(columns: List[str]) -> List[str]:
    out: List[str] = []
    for column in columns:
        key = normalize_colname(column)
        has_p = any(token in key for token in ["p_value", "p_val", "pval", "ttest", "p"]) 
        diet_count = sum(token in key for token in ["cd", "hfdl", "hfdo"])
        if has_p and diet_count >= 2:
            out.append(column)
    return out


def _select_phospho_qvalue_columns(columns: List[str]) -> List[str]:
    out: List[str] = []
    for column in columns:
        key = normalize_colname(column)
        has_q = any(token in key for token in ["qvalue", "q_value", "q_val", "fdr", "adj_p", "benjam"])
        diet_count = sum(token in key for token in ["cd", "hfdl", "hfdo"])
        if has_q and diet_count >= 2:
            out.append(column)
    return out


def _extract_diet_pair(column_name: str) -> Optional[tuple[str, str]]:
    hits: list[tuple[int, str]] = []
    upper = str(column_name).upper()
    for diet in ["CD", "HFDL", "HFDO"]:
        match = re.search(rf"(^|[^A-Z0-9]){diet}([^A-Z0-9]|$)", upper)
        if match:
            hits.append((match.start(), diet))
    hits.sort(key=lambda item: item[0])
    if len(hits) < 2:
        return None
    return hits[0][1], hits[1][1]


def _find_related_column(
    target_col: str,
    candidate_cols: List[str],
    required_tokens: List[str],
) -> Optional[str]:
    target_key = normalize_colname(target_col)
    pair = _extract_diet_pair(target_col)
    if pair is None:
        return None

    best: Optional[str] = None
    best_score = -1
    for candidate in candidate_cols:
        candidate_key = normalize_colname(candidate)
        if not all(token in candidate_key for token in required_tokens):
            continue
        pair_candidate = _extract_diet_pair(candidate)
        if pair_candidate != pair:
            continue

        score = 0
        for token in target_key.split("_"):
            if token and token in candidate_key:
                score += 1
        if score > best_score:
            best_score = score
            best = candidate
    return best


def parse_phospho_workbook_long(path: Path, pvalue_threshold: float = 0.05) -> pd.DataFrame:
    name = path.name.lower()
    if "faf107" in name:
        sheet_name = "Perseus Output"
        quantity_mode = "ptm"
    elif "faf108" in name:
        sheet_name = "Analysis"
        quantity_mode = "log"
    else:
        raise ValueError(f"Unsupported phospho workbook: {path.name}")

    raw = pd.read_excel(path, sheet_name=sheet_name)
    if "PG.Genes" not in raw.columns:
        raise ValueError(f"Missing PG.Genes in {path.name} ({sheet_name})")

    columns = [str(column) for column in raw.columns]
    quantity_cols = _select_phospho_quantity_columns(columns, quantity_mode)
    pvalue_cols = _select_phospho_pvalue_columns(columns)
    qvalue_cols = _select_phospho_qvalue_columns(columns)
    fc_cols = [
        column
        for column in columns
        if any(token in normalize_colname(column) for token in ["diff", "log2fc", "fold_change", "foldchange", "ratio", "fc"])
    ]

    if not pvalue_cols:
        raise ValueError(f"No phospho p-value columns found in {path.name} ({sheet_name})")

    numeric_cols = set(quantity_cols + pvalue_cols + qvalue_cols + fc_cols)
    for column in numeric_cols:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")

    sample_class = infer_sample_class_from_name(path.name)
    frames: List[pd.DataFrame] = []

    for p_col in pvalue_cols:
        pair = _extract_diet_pair(p_col)
        if pair is None:
            continue
        group_a, group_b = pair

        block = raw[raw[p_col] <= pvalue_threshold].copy()
        if block.empty:
            continue

        block["feature_name"] = block["PG.Genes"].astype(str).apply(_split_gene_tokens)
        block = block.explode("feature_name")
        block["feature_name"] = block["feature_name"].astype(str).str.strip()
        block = block[(block["feature_name"] != "") & (block["feature_name"].str.lower() != "nan")].copy()
        if block.empty:
            continue

        block = block.sort_values(p_col, na_position="last").drop_duplicates(subset=["feature_name"], keep="first")

        q_col = _find_related_column(p_col, qvalue_cols, required_tokens=[])
        fc_col = _find_related_column(p_col, fc_cols, required_tokens=[])

        avg_values = block[quantity_cols].mean(axis=1, skipna=True) if quantity_cols else pd.Series(pd.NA, index=block.index)
        med_values = block[quantity_cols].median(axis=1, skipna=True) if quantity_cols else pd.Series(pd.NA, index=block.index)
        foldchange = block[fc_col] if fc_col else pd.Series(pd.NA, index=block.index)

        frame = pd.DataFrame(
            {
                "feature_name": block["feature_name"].astype("string"),
                "accession": block["feature_name"].astype("string"),
                "comparison_id": f"proteomics:{path.stem}:{group_a}_vs_{group_b}",
                "group_a": group_a,
                "group_b": group_b,
                "log2fc_or_fc": pd.to_numeric(foldchange, errors="coerce"),
                "p_value": pd.to_numeric(block[p_col], errors="coerce"),
                "p_adj": pd.to_numeric(block[q_col], errors="coerce") if q_col else pd.Series(pd.NA, index=block.index),
                "direction": pd.NA,
                "avg_value": pd.to_numeric(avg_values, errors="coerce"),
                "median_value": pd.to_numeric(med_values, errors="coerce"),
                "source_dataset": "proteomics",
                "source_file": str(path),
                "modality": "proteomics",
                "sample_class": sample_class,
                "input_kind": f"excel_phospho_{sheet_name}",
                "celltype_label": "bulk",
                "manual_ct_lvl": pd.NA,
            }
        )

        frame.loc[frame["log2fc_or_fc"] > 0, "direction"] = "up"
        frame.loc[frame["log2fc_or_fc"] < 0, "direction"] = "down"
        frames.append(frame[REQUIRED_COLUMNS])

    if not frames:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def parse_bulk_rnaseq_file(path: Path) -> ParsedFileMetadata:
    comparison_id = f"organoid_bulk_rnaseq:{path.stem}"
    return ParsedFileMetadata(
        source_dataset="organoid_bulk_rnaseq",
        modality="bulk_rnaseq",
        input_kind=path.suffix.lower().lstrip("."),
        sample_class="bulk",
        source_file=str(path),
        comparison_id=comparison_id,
        group_a="NA",
        group_b="NA",
        celltype_label="bulk",
        manual_ct_lvl=None,
    )


def _first_existing_column(df: pd.DataFrame, candidates: List[str]) -> pd.Series:
    for candidate in candidates:
        if candidate in df.columns:
            return df[candidate]
    return pd.Series(pd.NA, index=df.index)


def _parse_bool_series(values: pd.Series) -> pd.Series:
    normalized = values.astype(str).str.strip().str.lower()
    return normalized.isin({"true", "t", "1", "yes"})


def load_rds_summarizedexperiment_rowdata(path: Path) -> pd.DataFrame:
    path = path.expanduser().resolve()
    r_code = """
args <- commandArgs(trailingOnly = TRUE)
input_path <- args[1]
output_path <- args[2]

suppressPackageStartupMessages({
  library(SummarizedExperiment)
  library(matrixStats)
})

obj <- readRDS(input_path)
if (!methods::is(obj, 'SummarizedExperiment')) {
  stop('Unsupported RDS class: expected SummarizedExperiment')
}

row_df <- as.data.frame(SummarizedExperiment::rowData(obj))
assay_mat <- as.matrix(SummarizedExperiment::assay(obj))
row_df$feature_rowname <- rownames(assay_mat)
row_df$avg_value <- rowMeans(assay_mat, na.rm = TRUE)
row_df$median_value <- matrixStats::rowMedians(assay_mat, na.rm = TRUE)

utils::write.csv(row_df, output_path, row.names = FALSE)
"""

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as handle:
        output_path = Path(handle.name)

    try:
        def pick_rscript() -> str:
            global _R_SCRIPT_CACHE
            if _R_SCRIPT_CACHE:
                return _R_SCRIPT_CACHE

            candidates: List[str] = []
            conda_prefix = os.environ.get("CONDA_PREFIX")
            if conda_prefix:
                candidates.append(str(Path(conda_prefix) / "bin" / "Rscript"))

            candidates.append(str(Path(sys.executable).resolve().parent / "Rscript"))

            which_rscript = shutil.which("Rscript")
            if which_rscript:
                candidates.append(which_rscript)

            candidates.append("/usr/local/bin/Rscript")

            seen = set()
            unique_candidates = []
            for candidate in candidates:
                if candidate not in seen:
                    seen.add(candidate)
                    unique_candidates.append(candidate)

            probe_code = (
                "suppressPackageStartupMessages(library(SummarizedExperiment)); "
                "suppressPackageStartupMessages(library(matrixStats));"
            )
            for candidate in unique_candidates:
                candidate_path = Path(candidate)
                if candidate != "Rscript" and not candidate_path.exists():
                    continue
                try:
                    subprocess.run(
                        [candidate, "-e", probe_code],
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    _R_SCRIPT_CACHE = candidate
                    return candidate
                except Exception:
                    continue

            raise RuntimeError(
                "No Rscript with required packages found. Install Bioconductor packages "
                "'SummarizedExperiment' and 'matrixStats' in the R runtime used by the notebook."
            )

        rscript = pick_rscript()
        cmd = ["Rscript", "-e", r_code, str(path), str(output_path)]
        cmd[0] = rscript
        try:
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            stdout = (exc.stdout or "").strip()
            details = stderr if stderr else stdout
            raise RuntimeError(
                f"R extraction failed for {path.name}: {details}"
            ) from exc
        return pd.read_csv(output_path)
    finally:
        output_path.unlink(missing_ok=True)


def parse_proteomics_rds_long(path: Path) -> pd.DataFrame:
    meta = parse_proteomics_file(path)
    raw = load_rds_summarizedexperiment_rowdata(path)

    diff_cols = [column for column in raw.columns if column.endswith("_diff")]
    comparison_stems = sorted(column[: -len("_diff")] for column in diff_cols)

    if not comparison_stems:
        raise ValueError(f"No comparison-specific diff columns found in {path.name}")

    frames: List[pd.DataFrame] = []
    feature_name = _first_existing_column(raw, ["Genes", "name", "feature_rowname", "ID"])
    accession = _first_existing_column(raw, ["Accession", "ID", "Ensembl_Gene_ID"])

    for stem in comparison_stems:
        group_tokens = stem.split("_vs_")
        group_a = group_tokens[0] if len(group_tokens) == 2 else "NA"
        group_b = group_tokens[1] if len(group_tokens) == 2 else "NA"

        frame = pd.DataFrame(
            {
                "feature_name": feature_name,
                "accession": accession,
                "comparison_id": f"{meta.source_dataset}:{meta.sample_class}:{stem}",
                "group_a": group_a,
                "group_b": group_b,
                "log2fc_or_fc": raw.get(f"{stem}_diff", pd.NA),
                "p_value": raw.get(f"{stem}_p.val", pd.NA),
                "p_adj": raw.get(f"{stem}_p.adj", pd.NA),
                "direction": pd.NA,
                "avg_value": raw.get("avg_value", pd.NA),
                "median_value": raw.get("median_value", pd.NA),
                "source_dataset": meta.source_dataset,
                "source_file": meta.source_file,
                "modality": meta.modality,
                "sample_class": meta.sample_class,
                "input_kind": "rds_summarizedexperiment",
                "celltype_label": meta.celltype_label,
                "manual_ct_lvl": pd.NA,
            }
        )

        significant_col = f"{stem}_significant"
        if significant_col in raw.columns:
            frame = frame[_parse_bool_series(raw[significant_col])].copy()

        frame["feature_name"] = frame["feature_name"].astype("string")
        frame = frame[frame["feature_name"].notna()]
        frame["log2fc_or_fc"] = pd.to_numeric(frame["log2fc_or_fc"], errors="coerce")
        frame["p_value"] = pd.to_numeric(frame["p_value"], errors="coerce")
        frame["p_adj"] = pd.to_numeric(frame["p_adj"], errors="coerce")
        frame["avg_value"] = pd.to_numeric(frame["avg_value"], errors="coerce")
        frame["median_value"] = pd.to_numeric(frame["median_value"], errors="coerce")
        frame.loc[frame["log2fc_or_fc"] > 0, "direction"] = "up"
        frame.loc[frame["log2fc_or_fc"] < 0, "direction"] = "down"
        frames.append(frame[REQUIRED_COLUMNS])

    return pd.concat(frames, ignore_index=True)


def normalize_table(df: pd.DataFrame, meta: ParsedFileMetadata) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    for canonical, aliases in COLUMN_ALIASES.items():
        src = first_matching_column(df, aliases)
        out[canonical] = df[src] if src else pd.NA

    out["source_dataset"] = meta.source_dataset
    out["source_file"] = meta.source_file
    out["modality"] = meta.modality
    out["sample_class"] = meta.sample_class
    out["input_kind"] = meta.input_kind
    out["comparison_id"] = meta.comparison_id

    out["group_a"] = out["group_a"].fillna(meta.group_a)
    out["group_b"] = out["group_b"].fillna(meta.group_b)
    out["celltype_label"] = meta.celltype_label
    out["manual_ct_lvl"] = meta.manual_ct_lvl if meta.manual_ct_lvl else pd.NA

    # If direction is missing, infer from FC sign when available.
    if out["direction"].isna().all():
        fc_numeric = pd.to_numeric(out["log2fc_or_fc"], errors="coerce")
        out.loc[fc_numeric > 0, "direction"] = "up"
        out.loc[fc_numeric < 0, "direction"] = "down"

    out["feature_name"] = out["feature_name"].astype("string")
    out = out[out["feature_name"].notna()]
    out["accession"] = out["accession"].fillna(out["feature_name"])

    for col in ["log2fc_or_fc", "p_value", "p_adj", "avg_value", "median_value"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    return out[REQUIRED_COLUMNS]


def enforce_schema_rules(df: pd.DataFrame) -> pd.DataFrame:
    if not set(df["modality"].dropna().unique()).issubset(VALID_MODALITIES):
        invalid = sorted(set(df["modality"].dropna().unique()) - VALID_MODALITIES)
        raise ValueError(f"Invalid modality values found: {invalid}")

    prot_or_bulk = df["modality"].isin(["proteomics", "bulk_rnaseq"])
    if not (df.loc[prot_or_bulk, "celltype_label"].fillna("bulk") == "bulk").all():
        raise ValueError("Proteomics/bulk_rnaseq rows must have celltype_label='bulk'.")

    tx = df["modality"] == "transcriptomics"
    missing_ct = df.loc[tx, "celltype_label"].isna().any()
    missing_lvl = df.loc[tx, "manual_ct_lvl"].isna().any()
    if missing_ct or missing_lvl:
        raise ValueError("Transcriptomics rows require both celltype_label and manual_ct_lvl.")

    return df


def load_excel_all_sheets(path: Path) -> List[pd.DataFrame]:
    xls = pd.ExcelFile(path)
    tables: List[pd.DataFrame] = []
    for sheet in xls.sheet_names:
        df = xls.parse(sheet)
        if df.shape[1] == 0 or df.shape[0] == 0:
            continue
        tables.append(df)
    return tables


def load_rds_tabular(path: Path) -> List[pd.DataFrame]:
    try:
        import pyreadr
    except ImportError as exc:
        raise ImportError(
            "Reading .rds files requires pyreadr. Install with `pip install pyreadr`."
        ) from exc

    result = pyreadr.read_r(str(path))
    tables: List[pd.DataFrame] = []
    for _, obj in result.items():
        if isinstance(obj, pd.DataFrame) and obj.shape[0] > 0 and obj.shape[1] > 0:
            tables.append(obj)
    return tables
