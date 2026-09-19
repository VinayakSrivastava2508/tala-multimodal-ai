"""Tests for src/reference_features.py (Day 2.6A chunking, entity extraction,
table/image extraction for the Multimodal Reference Package)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.reference_features import (
    build_chunks,
    chunk_section_body,
    classify_table_type,
    extract_entities,
    extract_html_images,
    extract_html_tables,
    is_decorative_image,
    split_into_sections,
)


# ── HTML boilerplate removal is delegated to hydration.extract_main_text
# (bs4/trafilatura strip nav/footer/script) -- reference_features operates on
# already-extracted text, so these tests cover chunking/entity behaviour on
# clean text instead of re-testing boilerplate stripping.

# ── section splitting ──────────────────────────────────────────────────────────

def test_split_into_sections_detects_page_markers():
    text = "[page 1]\nIntro text here.\n\n[page 2]\nMore text on page two."
    sections = split_into_sections(text)
    assert sections[0]["page_number"] == 1
    assert sections[1]["page_number"] == 2


def test_split_into_sections_falls_back_to_single_section_for_plain_text():
    text = "Just a plain paragraph with no headings or page markers at all."
    sections = split_into_sections(text)
    assert len(sections) == 1
    assert sections[0]["body"] == text


# ── chunking: target word count, no mid-sentence numeric splits, no empty chunks ─

def test_chunk_section_body_respects_target_word_range():
    body = " ".join(["word"] * 800)
    chunks = chunk_section_body(body, target_min=250, target_max=500)
    for c in chunks[:-1]:  # last chunk may be a remainder, shorter is fine
        assert len(c.split()) <= 500


def test_chunk_section_body_never_returns_empty_chunks():
    chunks = chunk_section_body("")
    assert all(c.strip() for c in chunks)
    assert chunks == []


def test_chunk_section_body_preserves_numerical_claim_with_unit_in_one_chunk():
    """A claim like '30% recycled polyester' must never be split so the number
    and its unit/qualifier land in different chunks."""
    sentence = "This garment is made from 30% recycled polyester and 70% organic cotton."
    body = ("Filler text. " * 5) + sentence + (" More filler text." * 5)
    chunks = chunk_section_body(body, target_min=10, target_max=40)
    found = [c for c in chunks if "30% recycled polyester" in c]
    assert len(found) == 1  # the whole claim phrase survives intact in exactly one chunk


def test_build_chunks_produces_provenance_on_every_chunk():
    text = "Our factories are audited annually. " * 20
    rows = build_chunks(
        reference_id="ref_tala_001", brand="TALA", reference_type="supplier_disclosure",
        document_title="Our Factories", extracted_text=text,
        source_url="https://www.wearetala.com/pages/our-factories",
        evidence_strength="strong", retrieved_at="2026-09-19T00:00:00Z",
    )
    assert len(rows) > 0
    for row in rows:
        assert row["reference_id"] == "ref_tala_001"
        assert row["source_url"] == "https://www.wearetala.com/pages/our-factories"
        assert row["chunk_text"].strip() != ""
        assert row["chunk_sequence"] >= 1


def test_build_chunks_uses_small_chunks_for_size_table_sections():
    """Part F: smaller chunks for tables/size charts/policy clauses -- a size
    section must not be forced into the 250-500 word prose target."""
    size_text = "UK 8: chest 82cm waist 64cm. UK 10: chest 86cm waist 68cm."
    text = f"Size Guide\n{size_text}"
    rows = build_chunks(
        reference_id="ref_tala_002", brand="TALA", reference_type="size_guide",
        document_title="Size Guide", extracted_text=text,
        source_url="https://www.wearetala.com/pages/size-guide",
        evidence_strength="strong", retrieved_at="2026-09-19T00:00:00Z",
    )
    assert len(rows) == 1
    assert "UK 8" in rows[0]["chunk_text"]


# ── structured entity extraction: numerical claims, units, materials, care ────

def test_extract_entities_captures_percentages():
    entities = extract_entities("Made from 65% recycled polyester and 35% elastane.")
    assert "65%" in entities["percentages"]
    assert "35%" in entities["percentages"]


def test_extract_entities_captures_quantities_with_units():
    entities = extract_entities("Weighs approximately 250 g and packaged in a 5 kg box.")
    assert any("250" in q and "g" in q.lower() for q in entities["quantities"])
    assert any("5" in q and "kg" in q.lower() for q in entities["quantities"])


def test_extract_entities_captures_target_years():
    entities = extract_entities("We aim to be net zero by 2030 and fully circular by 2035.")
    assert "2030" in entities["target_years"]
    assert "2035" in entities["target_years"]


def test_extract_entities_captures_known_materials():
    entities = extract_entities("This legging is made from recycled polyester and elastane.")
    assert "recycled polyester" in entities["materials"]
    assert "elastane" in entities["materials"]


def test_extract_entities_captures_certifications():
    entities = extract_entities("Our factories are OEKO-TEX certified and we are a certified B Corp.")
    assert "oeko-tex" in entities["certifications"]
    assert "b corp" in entities["certifications"]


def test_extract_entities_captures_care_instructions():
    entities = extract_entities("Machine wash cold. Do not tumble dry. Do not bleach.")
    assert "machine wash" in entities["care_instructions"]
    assert "do not tumble dry" in entities["care_instructions"]
    assert "do not bleach" in entities["care_instructions"]


def test_extract_entities_captures_return_window():
    entities = extract_entities("You may return your order within 30 days of purchase for a full refund.")
    assert any("30" in w for w in entities["return_windows"])


def test_extract_entities_does_not_fabricate_absent_entities():
    entities = extract_entities("This page has no numbers, materials, or certifications mentioned.")
    assert entities["percentages"] == []
    assert entities["materials"] == []
    assert entities["certifications"] == []


# ── table extraction: provenance, controlled table_type, size table ───────────

def test_extract_html_tables_captures_size_chart_with_provenance():
    html = """
    <html><body>
    <table>
      <tr><th>Size</th><th>Chest (cm)</th><th>Waist (cm)</th></tr>
      <tr><td>UK 8</td><td>82</td><td>64</td></tr>
      <tr><td>UK 10</td><td>86</td><td>68</td></tr>
    </table>
    </body></html>
    """
    rows = extract_html_tables(html, "ref_tala_003", "https://www.wearetala.com/pages/size-guide", "strong")
    assert len(rows) == 1
    assert rows[0]["table_type"] == "size_chart"
    assert rows[0]["source_url"] == "https://www.wearetala.com/pages/size-guide"
    assert rows[0]["reference_id"] == "ref_tala_003"
    assert "UK 8" in rows[0]["rows_or_normalised_text"]


def test_extract_html_tables_returns_empty_list_for_no_tables():
    assert extract_html_tables("<html><body><p>No tables here.</p></body></html>", "ref_x", "https://x", "strong") == []


def test_classify_table_type_returns_controlled_vocabulary():
    assert classify_table_type(["Size", "Chest", "Waist"]) == "size_chart"
    assert classify_table_type(["Material", "Composition %"]) == "material_composition"
    assert classify_table_type(["Random", "Columns"]) == "other"


# ── document images: decorative exclusion, certification retention ────────────

def test_is_decorative_image_excludes_icons_and_logos():
    assert is_decorative_image("https://cdn.example.com/icon-arrow.svg", "")
    assert is_decorative_image("https://cdn.example.com/nav-logo-small.png", "")


def test_is_decorative_image_retains_certification_badges():
    assert not is_decorative_image("https://cdn.example.com/oeko-tex-badge.png", "OEKO-TEX certified")
    assert not is_decorative_image("https://cdn.example.com/bcorp-logo.png", "B Corp certification")


def test_extract_html_images_excludes_decorative_but_keeps_certification_badge():
    html = """
    <html><body>
    <img src="https://cdn.example.com/nav-icon.svg" alt="menu icon">
    <img src="https://cdn.example.com/oeko-tex-badge.png" alt="OEKO-TEX certified">
    <img src="https://cdn.example.com/product-photo.jpg" alt="Product detail shot">
    </body></html>
    """
    rows = extract_html_images(html, "ref_tala_004", "https://www.wearetala.com/pages/sustainability", "strong")
    srcs = [r["source_url"] for r in rows]
    assert "https://cdn.example.com/nav-icon.svg" not in srcs
    assert "https://cdn.example.com/oeko-tex-badge.png" in srcs
    assert "https://cdn.example.com/product-photo.jpg" in srcs
    badge_row = next(r for r in rows if r["source_url"].endswith("oeko-tex-badge.png"))
    assert badge_row["image_role"] == "certification_badge"


# ── process_reference_package.py preserves manually-consolidated tables ───────

def test_process_reference_package_preserves_manual_consolidation_tables(tmp_path, monkeypatch):
    """Regression: re-running process_reference_package.py must never silently
    wipe a manually-built table (e.g. the Day 2.6B care-instruction
    applicability tables) that it has no way to regenerate itself."""
    import pandas as pd
    import scripts.merge_product_references_into_package as merge_mod
    import scripts.process_reference_package as mod

    manifest_path = tmp_path / "manifest.csv"
    chunks_path = tmp_path / "chunks.csv"
    tables_path = tmp_path / "tables.csv"
    images_path = tmp_path / "images.csv"

    pd.DataFrame([{
        "reference_id": "ref_1", "brand": "TALA", "reference_type": "sustainability_page",
        "document_title": "x", "extracted_text": "Some sustainability text. " * 20,
        "source_url": "https://example.com/x", "evidence_strength": "strong",
        "retrieved_at": "2026-01-01", "reference_status": "collected", "mime_type": "text/html",
        "has_tables": False,
    }]).to_csv(manifest_path, index=False)

    pd.DataFrame([{
        "table_id": "ref_care_001_table_01", "reference_id": "ref_care_001",
        "page_number_or_section": "", "table_type": "other", "headers": "a;b",
        "rows_or_normalised_text": "1;2", "units": "", "source_url": "",
        "extraction_method": "manual_consolidation", "evidence_strength": "strong",
        "provenance_note": "care-instruction-to-product applicability table",
    }]).to_csv(tables_path, index=False)

    monkeypatch.setattr(mod, "MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(mod, "CHUNKS_OUT", chunks_path)
    monkeypatch.setattr(mod, "TABLES_OUT", tables_path)
    monkeypatch.setattr(mod, "IMAGES_OUT", images_path)
    # This test is not about the Day 2.6B merge step -- point EXTRACTS_PATH at a
    # nonexistent file so process_reference_package.py's now-automatic merge
    # call is a documented no-op here, instead of silently falling through to
    # the real production merge module's own (unpatched) file paths.
    monkeypatch.setattr(merge_mod, "EXTRACTS_PATH", tmp_path / "does_not_exist.csv")

    mod.main()

    result = pd.read_csv(tables_path)
    assert "ref_care_001_table_01" in result["table_id"].values


def test_process_reference_package_reapplies_merge_when_extracts_present(tmp_path, monkeypatch):
    """Regression: process_reference_package.py must always re-apply the Day
    2.6B product-reference merge afterward, so rerunning it standalone can
    never again silently drop the care-applicability tables it can't
    regenerate itself (this previously required a human to remember to rerun
    merge_product_references_into_package.py separately)."""
    import pandas as pd
    import scripts.merge_product_references_into_package as merge_mod
    import scripts.process_reference_package as mod

    manifest_path = tmp_path / "manifest.csv"
    chunks_path = tmp_path / "chunks.csv"
    tables_path = tmp_path / "tables.csv"
    images_path = tmp_path / "images.csv"
    extracts_path = tmp_path / "extracts.csv"

    pd.DataFrame([{
        "reference_id": "ref_1", "brand": "TALA", "reference_type": "sustainability_page",
        "document_title": "x", "extracted_text": "Some sustainability text. " * 20,
        "source_url": "https://example.com/x", "evidence_strength": "strong",
        "retrieved_at": "2026-01-01", "reference_status": "collected", "mime_type": "text/html",
        "has_tables": False,
    }]).to_csv(manifest_path, index=False)

    pd.DataFrame([{
        "product_name": "Product A", "product_category": "leggings", "reference_subtype": "care_guidance",
        "care_temperature": "30C", "washing_method": "machine_wash", "drying_method": "line_dry",
        "ironing_guidance": "", "bleaching_guidance": "", "extracted_text": "Wash at 30C.",
        "source_url": "https://www.wearetala.com/products/a", "evidence_strength": "strong",
    }]).to_csv(extracts_path, index=False)

    monkeypatch.setattr(mod, "MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(mod, "CHUNKS_OUT", chunks_path)
    monkeypatch.setattr(mod, "TABLES_OUT", tables_path)
    monkeypatch.setattr(mod, "IMAGES_OUT", images_path)
    monkeypatch.setattr(merge_mod, "EXTRACTS_PATH", extracts_path)
    monkeypatch.setattr(merge_mod, "MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(merge_mod, "TABLES_OUT", tables_path)
    monkeypatch.setattr(merge_mod, "CARE_COVERAGE_OUT", tmp_path / "care_coverage.csv")

    mod.main()

    result = pd.read_csv(tables_path)
    assert (result["extraction_method"] == "manual_consolidation").sum() == 1
    merged_manifest = pd.read_csv(manifest_path)
    assert (merged_manifest["reference_type"] == "care_guide").sum() == 1


def test_process_reference_package_skips_merge_when_no_extracts(tmp_path, monkeypatch):
    """When no product-reference extracts exist yet, process_reference_package.py
    must complete successfully without attempting the merge."""
    import pandas as pd
    import scripts.merge_product_references_into_package as merge_mod
    import scripts.process_reference_package as mod

    manifest_path = tmp_path / "manifest.csv"
    chunks_path = tmp_path / "chunks.csv"
    tables_path = tmp_path / "tables.csv"
    images_path = tmp_path / "images.csv"

    pd.DataFrame([{
        "reference_id": "ref_1", "brand": "TALA", "reference_type": "sustainability_page",
        "document_title": "x", "extracted_text": "Some sustainability text. " * 20,
        "source_url": "https://example.com/x", "evidence_strength": "strong",
        "retrieved_at": "2026-01-01", "reference_status": "collected", "mime_type": "text/html",
        "has_tables": False,
    }]).to_csv(manifest_path, index=False)

    monkeypatch.setattr(mod, "MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(mod, "CHUNKS_OUT", chunks_path)
    monkeypatch.setattr(mod, "TABLES_OUT", tables_path)
    monkeypatch.setattr(mod, "IMAGES_OUT", images_path)
    monkeypatch.setattr(merge_mod, "EXTRACTS_PATH", tmp_path / "does_not_exist.csv")

    assert mod.main() == 0


# ── merge_product_references_into_package.py idempotency ──────────────────────

def test_merge_product_references_is_idempotent_on_manifest(tmp_path, monkeypatch):
    """Regression: rerunning the merge script (e.g. via the Day 2.6B
    orchestrator) once silently duplicated every product-derived reference row
    because it always appended onto the existing manifest rather than
    replacing its own prior output first."""
    import pandas as pd
    import scripts.merge_product_references_into_package as mod

    extracts_path = tmp_path / "extracts.csv"
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame([{
        "product_reference_id": "tpr_0001", "product_name": "Test Legging", "product_category": "leggings",
        "source_url": "https://www.wearetala.com/products/test-legging", "section_title": "Wear",
        "reference_subtype": "product_specification", "extracted_text": "A flattering legging. " * 10,
        "care_temperature": "", "washing_method": "", "drying_method": "", "ironing_guidance": "",
        "bleaching_guidance": "", "material_composition": "", "fit_guidance": "", "size_guidance": "",
        "extraction_method": "bs4_class_selector", "evidence_strength": "strong", "retrieved_at": "2026-01-01",
        "provenance_note": "product_handle=test-legging",
    }]).to_csv(extracts_path, index=False)
    pd.DataFrame(columns=["reference_id", "brand"]).to_csv(manifest_path, index=False)

    monkeypatch.setattr(mod, "EXTRACTS_PATH", extracts_path)
    monkeypatch.setattr(mod, "MANIFEST_PATH", manifest_path)

    mod.main()
    first_len = len(pd.read_csv(manifest_path))
    mod.main()  # rerun
    second_len = len(pd.read_csv(manifest_path))
    assert first_len == second_len  # must not double on rerun


def test_merge_product_references_dedupes_care_instructions_not_products(tmp_path, monkeypatch):
    """Identical care instructions across multiple products must consolidate
    into ONE reference row, not one per product."""
    import pandas as pd
    import scripts.merge_product_references_into_package as mod

    extracts_path = tmp_path / "extracts.csv"
    manifest_path = tmp_path / "manifest.csv"
    tables_path = tmp_path / "tables.csv"
    care_coverage_path = tmp_path / "care_coverage.csv"
    pd.DataFrame([
        {"product_reference_id": f"tpr_{i:04d}", "product_name": f"Cap {i}", "product_category": "accessories",
         "source_url": f"https://www.wearetala.com/products/cap-{i}", "section_title": "Care",
         "reference_subtype": "care_guidance", "extracted_text": "Do not wash. Do not tumble dry.",
         "care_temperature": "", "washing_method": "do_not_wash", "drying_method": "do_not_tumble_dry",
         "ironing_guidance": "", "bleaching_guidance": "", "material_composition": "", "fit_guidance": "",
         "size_guidance": "", "extraction_method": "bs4_class_selector", "evidence_strength": "strong",
         "retrieved_at": "2026-01-01", "provenance_note": f"product_handle=cap-{i}"}
        for i in range(3)
    ]).to_csv(extracts_path, index=False)
    pd.DataFrame(columns=["reference_id", "brand"]).to_csv(manifest_path, index=False)

    monkeypatch.setattr(mod, "EXTRACTS_PATH", extracts_path)
    monkeypatch.setattr(mod, "MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(mod, "TABLES_OUT", tables_path)
    monkeypatch.setattr(mod, "CARE_COVERAGE_OUT", care_coverage_path)
    mod.main()

    manifest = pd.read_csv(manifest_path)
    care_rows = manifest[manifest["reference_type"] == "care_guide"]
    assert len(care_rows) == 1  # one unique instruction, not 3 independent documents
    assert "Cap 0" in care_rows.iloc[0]["provenance_note"]
    assert "Cap 2" in care_rows.iloc[0]["provenance_note"]
