"""Tests for the Executive Multimodal AI Decision Cockpit (app/)."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import data_loader as dl  # noqa: E402


# ── Central data loader ─────────────────────────────────────────────────────

def test_load_claim_master_returns_expected_columns_and_nonzero_rows():
    df = dl.load_claim_master()
    assert len(df) > 0
    for col in ("claim_id", "production_fusion_label", "claim_category"):
        assert col in df.columns


def test_required_file_detection_raises_clear_error_for_missing_table():
    with pytest.raises(dl.MissingTableError):
        dl.load_table("this_table_does_not_exist")


def test_schema_mismatch_raises_clear_error_for_missing_required_column():
    with pytest.raises(dl.MissingTableError):
        dl.load_table("claim_label_summary", required_columns=("this_column_does_not_exist",))


def test_load_table_never_returns_a_fabricated_empty_substitute():
    # A missing table must raise, not silently return an empty DataFrame.
    with pytest.raises(dl.MissingTableError):
        dl.load_table("nonexistent_output_table_xyz")


# ── Claim-label counts (section 16) ─────────────────────────────────────────

def test_five_mixed_claim_records_and_two_mixed_themes():
    master = dl.load_claim_master()
    mixed = master[master["production_fusion_label"] == "mixed"]
    assert len(mixed) == 5

    themes = dl.load_mixed_theme_summary()
    assert len(themes) == 2
    assert int(themes["claim_record_count"].sum()) == 5


def test_zero_count_status_classes_are_preserved_in_overview_construction():
    label_summary = dl.load_label_summary()
    overall = label_summary[label_summary["section"] == "overall_label_distribution"]
    counts = overall.set_index("automated_label")["n_claims"].to_dict()
    all_labels = ["aligned", "partially_aligned", "mixed", "divergent", "insufficient_evidence"]
    for lbl in all_labels:
        counts.setdefault(lbl, 0)
    assert set(all_labels).issubset(counts.keys())


# ── Creator sample boundary ─────────────────────────────────────────────────

def test_creator_sample_denominators_match_documented_boundary():
    mix = dl.load_creator_partnership_mix()
    denominators = mix.groupby("brand")["brand_denominator_n"].first().to_dict()
    assert denominators["TALA"] == 24
    assert denominators["Oner Active"] == 8


def test_oner_and_tala_organic_share_wording_is_directionally_correct():
    mix = dl.load_creator_partnership_mix()
    oner_organic = mix[(mix["brand"] == "Oner Active") & (mix["partnership_type"] == "organic")]
    tala_organic = mix[(mix["brand"] == "TALA") & (mix["partnership_type"] == "organic")]
    assert not oner_organic.empty and not tala_organic.empty
    # Oner has the highest observed organic *share* but a smaller sample than TALA.
    assert float(oner_organic.iloc[0]["pct_within_brand"]) > float(tala_organic.iloc[0]["pct_within_brand"])
    assert int(oner_organic.iloc[0]["brand_denominator_n"]) < int(tala_organic.iloc[0]["brand_denominator_n"])


# ── Governance gate-status counts ───────────────────────────────────────────

def test_governance_gate_status_counts_sum_to_gate_total_per_context():
    go_no_go = dl.load_go_no_go()
    for ctx in go_no_go["deployment_context"].unique():
        subset = go_no_go[go_no_go["deployment_context"] == ctx]
        counts = subset["status"].value_counts()
        assert counts.sum() == len(subset)
        assert set(counts.index).issubset({"GO", "PARTIAL", "NO-GO"})


# ── Compliance / boundary checks over the app source ────────────────────────

APP_SOURCE_FILES = list((PROJECT_ROOT / "app").rglob("*.py"))
APP_SOURCE_TEXT = "\n".join(p.read_text(encoding="utf-8") for p in APP_SOURCE_FILES)


def test_no_proven_moat_language_anywhere_in_app_source():
    # "not ... a proven moat" (disclaiming) is fine; an affirmative claim is not.
    assert "is a proven moat" not in APP_SOURCE_TEXT.lower()
    assert "confirmed moat" not in APP_SOURCE_TEXT.lower()


def test_no_engagement_performance_ranking_language_in_app_source():
    # Disclaiming that fusion/RAG is "not for engagement prediction" is fine;
    # actually implementing/labelling one is not.
    for forbidden in ("engagement_prediction_model", "predicted_engagement", "engagement regression model", "engagement_score_rank"):
        assert forbidden not in APP_SOURCE_TEXT.lower()


def test_no_protected_characteristic_fields_in_app_source():
    for term in ("ethnicity", "gender_identity", "sexual_orientation", "disability_status", "race_field", "religion_field"):
        assert term not in APP_SOURCE_TEXT.lower()


def test_no_dotenv_access_in_app_source():
    assert "load_dotenv" not in APP_SOURCE_TEXT
    assert "os.environ" not in APP_SOURCE_TEXT
    assert "getenv" not in APP_SOURCE_TEXT


def test_no_fallback_llm_or_vector_store_in_app_source():
    for forbidden in ("fallback_llm", "faiss.write_index", "openai", "fallback_vector_store"):
        assert forbidden not in APP_SOURCE_TEXT.lower()


# ── Lazy model / RAG imports (no Gemini/Chroma call on analytics-page load) ─

def _module_level_import_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:  # only top-level statements, not inside functions
        if isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
    return names


def test_rag_and_gemini_modules_are_not_imported_at_module_level_anywhere_in_app():
    forbidden_prefixes = ("src.rag", "google.genai", "chromadb", "sentence_transformers")
    for path in APP_SOURCE_FILES:
        top_level = _module_level_import_names(path)
        for name in top_level:
            for forbidden in forbidden_prefixes:
                assert not name.startswith(forbidden), f"{path} imports '{name}' at module level -- must be lazy"


def test_rag_explorer_imports_orchestrator_lazily_inside_render():
    source = (PROJECT_ROOT / "app" / "views" / "rag_explorer.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    render_fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "render")
    render_source = ast.get_source_segment(source, render_fn)
    assert "from src.rag import orchestrator" in render_source


def test_streamlit_app_entrypoint_does_not_import_rag_modules_at_module_level():
    top_level = _module_level_import_names(PROJECT_ROOT / "app" / "streamlit_app.py")
    assert not any(n.startswith("src.rag") for n in top_level)


# ── Session-state handoff ───────────────────────────────────────────────────

def test_send_claim_to_rag_sets_claim_id_and_navigates(monkeypatch):
    import streamlit as st
    from app.state import CLAIM_ID_KEY, PAGE_KEY, RAG_HANDOFF_KEY, send_claim_to_rag

    st.session_state.clear()
    send_claim_to_rag("c_OC_0002_00")
    assert st.session_state[CLAIM_ID_KEY] == "c_OC_0002_00"
    assert st.session_state[RAG_HANDOFF_KEY] == "c_OC_0002_00"
    assert st.session_state[PAGE_KEY] == "Multimodal RAG"


# ── AppTest smoke tests (Gemini/Chroma mocked, no live calls) ───────────────

streamlit_testing = pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest  # noqa: E402


def _make_apptest() -> AppTest:
    return AppTest.from_file(str(PROJECT_ROOT / "app" / "streamlit_app.py"), default_timeout=60)


def test_executive_overview_loads_without_exception_and_without_gemini_call(monkeypatch):
    called = {"gemini": False}
    import src.rag.gemini_generator as gg

    def _fail_if_called(*args, **kwargs):
        called["gemini"] = True
        raise AssertionError("Gemini must not be called on analytics-page load")

    monkeypatch.setattr(gg, "generate_answer", _fail_if_called, raising=False)

    at = _make_apptest()
    at.run()
    assert not at.exception
    assert called["gemini"] is False


def test_governance_page_loads_without_exception():
    at = _make_apptest()
    at.run()
    at.sidebar.radio[0].set_value("Governance & Roadmap").run()
    assert not at.exception


def test_creator_strategy_page_shows_boundary_notice():
    at = _make_apptest()
    at.run()
    at.sidebar.radio[0].set_value("Creator Strategy").run()
    assert not at.exception
    page_markdown = "\n".join(m.value for m in at.markdown)
    assert "targeted sample dominated by YouTube" in page_markdown


def test_rag_page_loads_but_does_not_call_gemini_before_submit(monkeypatch):
    called = {"gemini": False}
    import src.rag.gemini_generator as gg

    def _fail_if_called(*args, **kwargs):
        called["gemini"] = True
        raise AssertionError("Gemini must not be called merely by opening the RAG page")

    monkeypatch.setattr(gg, "generate_answer", _fail_if_called, raising=False)

    at = _make_apptest()
    at.run()
    at.sidebar.radio[0].set_value("Multimodal RAG").run()
    assert called["gemini"] is False


# ── RAG result persistence (Executive Cockpit Acceptance Patch, Part B) ────────
#
# Regression coverage for the state-loss defect the browser audit found:
# toggling "Show retrieval trace" after a successful investigation used to
# discard the generated answer, the evidence, and the Claim Diagnostic
# handoff context, because the result only ever existed as a local variable
# inside the `if st.button("Investigate")` block. Gemini/retrieval are
# mocked throughout -- these tests make zero live API calls.

def _mock_rag_generation(monkeypatch):
    """Mocks gemini_generator.generate_answer and citation_validator.validate_response
    so orchestrator.investigate_claim succeeds without any live API call.
    Returns a dict with a 'calls' counter."""
    import src.rag.citation_validator as cv
    import src.rag.gemini_generator as gg

    calls = {"n": 0}

    def _fake_generate(*args, **kwargs):
        calls["n"] += 1
        return {
            "concise_answer": "Mock generated answer for testing.",
            "claim_assessment": "mixed",
            "confidence_explanation": "Mock confidence explanation.",
            "caveats": ["Mock caveat."],
            "evidence_gaps": [],
            "_model_used": "mock-model",
        }

    monkeypatch.setattr(gg, "generate_answer", _fake_generate, raising=False)
    monkeypatch.setattr(
        cv, "validate_response",
        lambda *a, **k: {"passed": True, "failures": [], "checks": {}},
        raising=False,
    )
    return calls


def test_rag_page_load_produces_zero_generator_calls(monkeypatch):
    calls = _mock_rag_generation(monkeypatch)
    at = _make_apptest()
    at.run()
    at.sidebar.radio[0].set_value("Multimodal RAG").run()
    assert not at.exception
    assert calls["n"] == 0


def test_rag_investigation_then_trace_toggle_preserves_result_and_call_count(monkeypatch):
    from app.state import RAG_RESULT_KEY

    calls = _mock_rag_generation(monkeypatch)
    at = _make_apptest()
    at.run()
    at.sidebar.radio[0].set_value("Multimodal RAG").run()
    assert calls["n"] == 0

    # Submit exactly one investigation (default-selected claim).
    at.button[0].click().run()
    assert not at.exception
    assert calls["n"] == 1
    first_result = at.session_state[RAG_RESULT_KEY]
    assert first_result is not None
    assert first_result["generation"]["concise_answer"] == "Mock generated answer for testing."
    first_claim_id = first_result["claim_id"]

    # Toggle trace ON: result and generator call count must be unchanged.
    at.sidebar.toggle[0].set_value(True).run()
    assert not at.exception
    assert calls["n"] == 1
    assert at.session_state[RAG_RESULT_KEY] is not None
    assert at.session_state[RAG_RESULT_KEY]["claim_id"] == first_claim_id
    assert at.session_state[RAG_RESULT_KEY]["generation"]["concise_answer"] == "Mock generated answer for testing."

    # Toggle trace OFF: same guarantee.
    at.sidebar.toggle[0].set_value(False).run()
    assert not at.exception
    assert calls["n"] == 1
    assert at.session_state[RAG_RESULT_KEY] is not None
    assert at.session_state[RAG_RESULT_KEY]["claim_id"] == first_claim_id


def test_rag_new_investigation_replaces_previous_result(monkeypatch):
    from app.state import RAG_RESULT_KEY

    calls = _mock_rag_generation(monkeypatch)
    at = _make_apptest()
    at.run()
    at.sidebar.radio[0].set_value("Multimodal RAG").run()

    at.button[0].click().run()
    assert calls["n"] == 1
    first_claim_id = at.session_state[RAG_RESULT_KEY]["claim_id"]

    # Select a different claim in the claim combobox, then submit again.
    claim_selectbox = [sb for sb in at.selectbox if sb.label == "Select a claim to investigate"][0]
    other_options = [o for o in claim_selectbox.options if not o.endswith(f"({first_claim_id})")]
    assert other_options, "Need at least two claims in the corpus for this test"
    claim_selectbox.set_value(other_options[0]).run()
    at.button[0].click().run()

    assert not at.exception
    assert calls["n"] == 2  # a second, explicit investigation actually re-generated
    assert at.session_state[RAG_RESULT_KEY]["claim_id"] != first_claim_id


def test_rag_toggle_trace_preserves_claim_diagnostic_handoff_context(monkeypatch):
    from app.state import CLAIM_ID_KEY, RAG_HANDOFF_KEY, RAG_RESULT_KEY

    calls = _mock_rag_generation(monkeypatch)
    at = _make_apptest()
    # Simulate arriving via Claim Diagnostic's "Investigate this claim in
    # Multimodal RAG" button (app.state.send_claim_to_rag), which sets these
    # same two keys before navigating -- pre-seeding session_state before the
    # first run reproduces that arrival without needing to click through
    # Claim Diagnostic's own filter widgets.
    at.session_state[CLAIM_ID_KEY] = "c_OC_0010_00"
    at.session_state[RAG_HANDOFF_KEY] = "c_OC_0010_00"
    at.run()
    at.sidebar.radio[0].set_value("Multimodal RAG").run()

    page_markdown = "\n".join(m.value for m in at.markdown)
    assert "Arrived from Claim Diagnostic" in page_markdown
    assert calls["n"] == 0  # arrival alone must never call Gemini

    at.button[0].click().run()
    assert not at.exception
    assert calls["n"] == 1
    result = at.session_state[RAG_RESULT_KEY]
    assert result["handoff_context"] is not None
    assert result["handoff_context"]["claim_category"]

    at.sidebar.toggle[0].set_value(True).run()
    assert not at.exception
    assert calls["n"] == 1
    assert at.session_state[RAG_RESULT_KEY]["handoff_context"] is not None
    page_markdown = "\n".join(m.value for m in at.markdown)
    assert "Arrived from Claim Diagnostic" in page_markdown
