from __future__ import annotations

import pandas as pd


def add_graph_context_features(query: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Build label-free context graph statistics from training/reference rows only.

    Nodes are order snapshots, styles, and production lines. Edges represent shared style,
    line, or style-line context. This is a relational graph baseline, not a GNN.
    """
    q = query.copy()
    ref = reference.copy()
    for col in ("style_category", "production_line"):
        if col not in q:
            q[col] = "UNKNOWN"
        if col not in ref:
            ref[col] = "UNKNOWN"
        q[col] = q[col].fillna("UNKNOWN").astype(str)
        ref[col] = ref[col].fillna("UNKNOWN").astype(str)
    style_counts = ref.groupby("style_category").size()
    line_counts = ref.groupby("production_line").size()
    pair_counts = ref.groupby(["style_category", "production_line"]).size()
    q["graph_style_degree"] = q.style_category.map(style_counts).fillna(0).astype(float)
    q["graph_line_degree"] = q.production_line.map(line_counts).fillna(0).astype(float)
    q["graph_style_line_support"] = [float(pair_counts.get((s, l), 0)) for s, l in zip(q.style_category, q.production_line)]
    q["graph_context_coverage"] = (q.graph_style_degree.gt(0).astype(int) + q.graph_line_degree.gt(0).astype(int) + q.graph_style_line_support.gt(0).astype(int)).astype(float) / 3
    return q
