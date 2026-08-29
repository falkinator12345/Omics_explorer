# Omics Explorer

Streamlit app for interactively browsing integrated multi-omics (snRNAseq + proteomics)
findings from the pig gut dataset. Published as the public, deployable companion to a
private development repository.

## What's in this repository
- `app.py` and `webapp/` — the Streamlit app
- `omics_overview/` — parsing/schema logic used to build the consolidated findings table
- `Data/` and `consolidated_significant_findings.parquet` — the processed data the app reads
  (large files tracked via Git LFS)
- `ressources/20260225_Color_scheme_Regional.txt` — plotting color scheme

Development-only material (raw source notebooks, the dataset-build CLI, environment dumps,
benchmark artifacts, and proteomics `.rds` sources) lives in the private working repository
and isn't published here; the proteomics `*_expression_cache.csv` files under
`Data/Proteomics/` are the ones the app actually reads.

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## Deployed app
<!-- Add the live Streamlit Community Cloud URL here once deployed -->
