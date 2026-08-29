"""Utilities to build and query a consolidated multi-omics significance dataset."""

from .build_dataset import build_consolidated_dataset
from .query import (
    filter_significant,
    find_feature,
    list_comparisons,
    overlap_features,
)
from .visualize import (
    load_aggregated_expression,
    plot_aggregated_dotplot,
    plot_aggregated_heatmap,
)

__all__ = [
    "build_consolidated_dataset",
    "find_feature",
    "list_comparisons",
    "overlap_features",
    "filter_significant",
    "load_aggregated_expression",
    "plot_aggregated_heatmap",
    "plot_aggregated_dotplot",
]
