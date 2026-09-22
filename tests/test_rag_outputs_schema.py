"""Confirms the Day 3B RAG output tables validate against their
configs/schema.yaml entries -- skipped if the pipeline hasn't been run."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.validation import guess_schema, validate

TABLES = PROJECT_ROOT / "outputs" / "tables"

RAG_TABLES = [
    "rag_collection_summary", "rag_corpus_coverage", "rag_modality_coverage",
    "rag_retrieval_evaluation", "rag_query_routing_evaluation", "rag_citation_validation",
    "rag_latency_summary", "rag_demo_cases", "rag_go_no_go",
]


@pytest.mark.parametrize("table_name", RAG_TABLES)
def test_rag_output_table_matches_its_schema(table_name):
    path = TABLES / f"{table_name}.csv"
    if not path.exists():
        pytest.skip(f"{table_name}.csv not generated in this environment")
    assert guess_schema(path.name) == table_name
    df = pd.read_csv(path)
    if df.empty:
        pytest.skip(f"{table_name}.csv is empty in this environment")
    passed, errors = validate(df, table_name)
    assert passed, errors
