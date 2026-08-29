# Methods paragraph — multi-omics explorer web application

## Version A — single paragraph (concise)

To enable integrated interrogation of the transcriptomic and proteomic datasets generated in this
work, a local web application was implemented in Python (v3.14) using Streamlit (v1.55). Because the
individual analyses were produced by heterogeneous pipelines, all results were first normalised into
a common long-format schema in which each row represents one feature (gene or protein) in one
pairwise contrast, annotated with effect size, nominal and adjusted p-value, direction of regulation,
mean and median abundance, modality, cell-type label and cell-type annotation level. Differential
gene expression results from the regional and high-fat-diet (HFD) snRNAseq datasets, derived from
pseudobulk edgeR testing on CellBender-corrected counts across pseudo-replicates (minimum cluster
size 100 cells, minimum expressed fraction 0.01, |log2FC| ≥ 0.5, Benjamini–Hochberg-adjusted
p ≤ 0.05), were parsed together with the ECM, soluble, villus and phosphoproteome results; the latter
were extracted from `SummarizedExperiment` objects via an embedded R bridge, retaining features
flagged as significant for each contrast, while phosphosite tables were filtered at p ≤ 0.05 and
resolved to gene level by retaining the most significant site per gene. The consolidated table
(3.82 × 10^6 rows, 325 contrasts, 17,973 unique features) was stored as a columnar Parquet file and
cached in memory at application start. The interface exposes three modules: (i) a query module
returning all significant detections of a user-supplied gene list across modalities, including the
condition in which each feature is enriched; (ii) an overlap module that computes exclusive
intersections between any combination of contrasts and visualises them as an UpSet-style plot with
optional log2 scaling, from which intersecting feature sets can be inspected and exported; and
(iii) a plotting module rendering aggregated expression as heatmaps or dot plots, either combined or
faceted per dataset. For expression-level visualisation, per-cell-type mean expression and percentage
of expressing nuclei were pre-computed from SCTransform log-normalised, CellBender-corrected counts
across all annotation levels (lvl0–lvl4), resolved by intestinal segment for the regional dataset and
by tissue and diet for the HFD dataset. As datasets differ in dynamic range, values were z-scored
independently within each dataset prior to joint visualisation, while all exports retain the
untransformed values; column annotation bars and categorical colours were drawn from the project-wide
colour scheme to keep figures consistent with the remainder of this work.

---

## Version B — split into two paragraphs (data integration / application)

**Data integration.** To enable integrated interrogation of the transcriptomic and proteomic datasets
generated in this work, all analysis results were consolidated into a single queryable resource.
Because the individual analyses were produced by heterogeneous pipelines, results were normalised into
a common long-format schema in which each row represents one feature (gene or protein) in one pairwise
contrast, annotated with effect size, nominal and adjusted p-value, direction of regulation, mean and
median abundance, modality, cell-type label and cell-type annotation level. Differential gene
expression results from the regional and high-fat-diet (HFD) snRNAseq datasets, derived from pseudobulk
edgeR testing on CellBender-corrected counts across pseudo-replicates (minimum cluster size 100 cells,
minimum expressed fraction 0.01, |log2FC| ≥ 0.5, Benjamini–Hochberg-adjusted p ≤ 0.05), were parsed
together with the ECM, soluble, villus and phosphoproteome results. Proteomic results were extracted
from `SummarizedExperiment` objects via an embedded R bridge, retaining features flagged as significant
for each contrast; phosphosite tables were filtered at p ≤ 0.05 and resolved to gene level by retaining
the most significant site per gene. The consolidated table comprised 3.82 × 10^6 rows covering 325
contrasts and 17,973 unique features and was stored as a columnar Parquet file.

**Application.** The resource is served by a local web application implemented in Python (v3.14) using
Streamlit (v1.55), which loads the consolidated table into memory at start-up and exposes three
modules. A query module returns all significant detections of a user-supplied gene list across
modalities, including the condition in which each feature is enriched. An overlap module computes
exclusive intersections between any combination of contrasts and visualises them as an UpSet-style
plot with optional log2 scaling; intersecting feature sets can be inspected and exported in long or
wide format. A plotting module renders aggregated expression as heatmaps or dot plots, either combined
across datasets or faceted per dataset. For expression-level visualisation, per-cell-type mean
expression and percentage of expressing nuclei were pre-computed from SCTransform log-normalised,
CellBender-corrected counts across all annotation levels (lvl0–lvl4), resolved by intestinal segment
for the regional dataset and by tissue and diet for the HFD dataset, allowing columns to be grouped
either by cell type or by condition. As datasets differ in dynamic range, values were z-scored
independently within each dataset prior to joint visualisation, while all exports retain the
untransformed values. Column annotation bars and categorical colours were drawn from the project-wide
colour scheme to keep figures consistent with the remainder of this work.

---

## Notes / placeholders to check before submission

- **Software versions** taken from the project environment (`scverse2026_mac.yml`): Python 3.14.3,
  Streamlit 1.55.0, pandas 2.3.3, pyarrow 23.0.0, scanpy 1.11.5, anndata 0.12.10. Confirm these match
  the environment you actually cite elsewhere in the thesis.
- **Dataset numbers** (3.82 × 10^6 rows / 325 contrasts / 17,973 features) reflect the current build;
  regenerate and update if further data is added.
- **Organoid bulk RNAseq** is deliberately not mentioned — the sequencing results were not yet
  available at the time of writing. The application already supports the modality, so add a clause
  once the data is integrated.
- **Citations to add**: edgeR, DElegate, CellBender, SCTransform, Streamlit, Apache Arrow/Parquet,
  and the UpSet visualisation concept (Lex et al., 2014), depending on your citation conventions.
- Adjust British/American spelling ("normalised" vs. "normalized") to match your thesis style.
