from __future__ import annotations

import pandas as pd


def add_knowledge_graph_features(query: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Materialize decision-time order/style/line/resource facts as compact KG features.

    Triples are represented as typed context keys and cohort support counts. No outcome
    labels are traversed, and reference rows must be limited to the training period.
    """
    q = query.copy()
    ref = reference.copy()
    categorical = [c for c in ["style_category", "production_line"] if c in q.columns]
    for col in categorical:
        q[col] = q[col].fillna("UNKNOWN").astype(str)
        ref[col] = ref[col].fillna("UNKNOWN").astype(str)
    if "style_category" in categorical:
        counts = ref.groupby("style_category").size()
        q["kg_style_fact_support"] = q.style_category.map(counts).fillna(0).astype(float)
    else:
        q["kg_style_fact_support"] = 0.0
    if "production_line" in categorical:
        counts = ref.groupby("production_line").size()
        q["kg_line_fact_support"] = q.production_line.map(counts).fillna(0).astype(float)
    else:
        q["kg_line_fact_support"] = 0.0
    if all(c in q for c in ["style_category", "production_line"]):
        triples = ref.groupby(["style_category", "production_line"]).size()
        q["kg_style_on_line_support"] = [float(triples.get((s, l), 0)) for s, l in zip(q.style_category, q.production_line)]
    else:
        q["kg_style_on_line_support"] = 0.0
    numeric = [c for c in ["available_machine_count", "required_machine_count"] if c in q]
    if len(numeric) == 2:
        q["kg_machine_coverage_ratio"] = (pd.to_numeric(q[numeric[0]], errors="coerce") /
                                          pd.to_numeric(q[numeric[1]], errors="coerce").replace(0, pd.NA)).fillna(0).clip(0, 5)
    else:
        q["kg_machine_coverage_ratio"] = 0.0
    available = [c for c in q.columns if c.startswith("available_") and c not in {"available_machine_count"}]
    committed = [c for c in q.columns if c.startswith("assigned_elsewhere_")]
    if available and committed:
        free, reserved = q[available].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1), q[committed].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)
        q["kg_workforce_reservation_ratio"] = reserved / (free + reserved).replace(0, pd.NA)
        q["kg_workforce_reservation_ratio"] = q["kg_workforce_reservation_ratio"].fillna(0).clip(0, 1)
    else:
        q["kg_workforce_reservation_ratio"] = 0.0
    return q
