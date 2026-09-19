"""Day 3A: primary multimodal evidence fusion for TALA claim-experience
divergence. Builds claim_evidence_units.csv, runs decision-level fusion in
four incremental modality configurations, and writes all comparison/
divergence/contribution tables and figures.

All results are automated, provisional, and NOT independently human-validated
-- see docs/automated_fusion_evaluation.md. No manual labelling of any kind is
performed anywhere in this pipeline.

Usage: python scripts/run_primary_multimodal_fusion.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.fusion.evidence_fusion import ALL_MODALITIES, fuse_all_claims  # noqa: E402
from src.fusion.image_evidence import build_image_evidence_unit  # noqa: E402
from src.fusion.reference_evidence import classify_reference_evidence, unsupported_reference_result  # noqa: E402
from src.fusion.text_evidence import (  # noqa: E402
    extract_passage,
    reliability_score,
    rule_based_stance,
    source_independence_for_corpus,
)
from src.fusion.video_evidence import build_video_evidence_unit  # noqa: E402
from scripts.build_claim_multimodal_candidates import load_text_corpora, match_text_evidence  # noqa: E402

CORPORA = PROJECT_ROOT / "data" / "corpora"
PROCESSED = PROJECT_ROOT / "data" / "processed"
FUSION_DIR = PROCESSED / "fusion"
INTERIM_DAY2_5 = PROJECT_ROOT / "data" / "interim" / "day2_5"
INTERIM_DAY2_6 = PROJECT_ROOT / "data" / "interim" / "day2_6"
TABLES = PROJECT_ROOT / "outputs" / "tables"
FIGURES = PROJECT_ROOT / "outputs" / "figures"

CONFIGURATIONS = [
    ("text_only", ["text"]),
    ("text_image", ["text", "image"]),
    ("text_image_video", ["text", "image", "video"]),
    ("text_image_video_reference", ["text", "image", "video", "reference"]),
]

EVIDENCE_UNIT_COLUMNS = [
    "evidence_unit_id", "claim_id", "claim_text", "claim_category", "product_name", "product_category",
    "modality", "evidence_id", "evidence_text", "evidence_passage", "source_type", "source_url",
    "source_date", "source_independence", "evidence_strength", "relevance_score", "relevance_method",
    "stance", "stance_score", "reliability_score", "visually_groundable", "visual_gate_passed",
    "visual_gate_reason", "permitted_evidence_role", "brand_evidence_eligible", "evaluation_status",
    "provisional_finding", "external_validation_recommended", "provenance_note",
]


def load_config() -> dict:
    with open(PROJECT_ROOT / "configs" / "fusion_rules.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_text_units(claims: pd.DataFrame, config: dict) -> list[dict]:
    """Text-evidence units from customer-experience + creator-strategy corpus
    matches (reusing the exact match_text_evidence logic already used to build
    claim_multimodal_evidence_candidates.csv, for a consistent per-document
    relevance score)."""
    corpus = load_text_corpora()
    threshold = config["relevance_thresholds"]["text_relevance_min"]
    text_matches = match_text_evidence(claims, corpus)

    units = []
    seq = 0
    for _, claim in claims.iterrows():
        matches = text_matches.get(claim["claim_id"], [])
        for doc_id, score in matches:
            doc_rows = corpus[corpus["document_id"] == doc_id]
            if doc_rows.empty:
                continue
            doc = doc_rows.iloc[0]
            corpus_source = doc.get("corpus_source", "")
            source_independence = source_independence_for_corpus(corpus_source)
            stance_result = rule_based_stance(doc["extracted_text"], score, threshold, config)
            strength = "strong" if score >= 0.6 else "medium" if score >= threshold else "weak"
            seq += 1
            units.append({
                "evidence_unit_id": f"eu_{seq:04d}", "claim_id": claim["claim_id"],
                "claim_text": claim["claim_text"], "claim_category": claim["claim_category"],
                "product_name": claim.get("product_name", ""), "product_category": claim.get("product_category", ""),
                "modality": "text", "evidence_id": doc_id, "evidence_text": doc["extracted_text"],
                "evidence_passage": extract_passage(doc["extracted_text"]),
                "source_type": "customer_review" if corpus_source == "customer_experience" else "press",
                "source_url": doc.get("source_url", ""), "source_date": doc.get("publication_date", ""),
                "source_independence": source_independence, "evidence_strength": strength,
                "relevance_score": score, "relevance_method": "sentence_transformers_all_MiniLM_L6_v2",
                "stance": stance_result["stance"], "stance_score": stance_result["stance_score"],
                "reliability_score": reliability_score(strength, source_independence, config),
                "visually_groundable": False, "visual_gate_passed": False, "visual_gate_reason": "not applicable to text",
                "permitted_evidence_role": "supports_or_challenges" if stance_result["stance"] in ("supports", "challenges") else "context_only",
                "brand_evidence_eligible": True,
                "evaluation_status": "automated_not_human_validated", "provisional_finding": True,
                "external_validation_recommended": stance_result["stance"] == "challenges",
                "provenance_note": f"matched via sentence-embedding cosine similarity {score} (threshold {threshold})",
            })
    return units


def build_official_disclosure_unit(claim: pd.Series, seq: int, config: dict) -> dict:
    """Part 6 rule 1: the original official claim itself can never validate
    itself -- it is recorded (for provenance/coverage reporting) but with
    stance='not_applicable' so it can never enter fuse_claim's eligible set."""
    return {
        "evidence_unit_id": f"eu_{seq:04d}", "claim_id": claim["claim_id"], "claim_text": claim["claim_text"],
        "claim_category": claim["claim_category"], "product_name": claim.get("product_name", ""),
        "product_category": claim.get("product_category", ""), "modality": "text",
        "evidence_id": claim["claim_id"], "evidence_text": claim["claim_text"],
        "evidence_passage": extract_passage(claim["claim_text"]), "source_type": "official_claim",
        "source_url": "", "source_date": "", "source_independence": "self_reported",
        "evidence_strength": "strong", "relevance_score": 1.0, "relevance_method": "identity",
        "stance": "not_applicable", "stance_score": 0.0, "reliability_score": 0.0,
        "visually_groundable": False, "visual_gate_passed": False,
        "visual_gate_reason": "the claim cannot be used as evidence for itself",
        "permitted_evidence_role": "excluded_self_validation", "brand_evidence_eligible": False,
        "evaluation_status": "automated_not_human_validated", "provisional_finding": True,
        "external_validation_recommended": False,
        "provenance_note": "Part 6 rule 1: original official claim, excluded from self-validation",
    }


def build_image_units(claims: pd.DataFrame, config: dict) -> list[dict]:
    image_matches = pd.read_csv(PROCESSED / "claim_visual_evidence_matches.csv") if (PROCESSED / "claim_visual_evidence_matches.csv").exists() else pd.DataFrame()
    image_assets = pd.read_csv(INTERIM_DAY2_5 / "image_assets.csv") if (INTERIM_DAY2_5 / "image_assets.csv").exists() else pd.DataFrame()
    if image_matches.empty:
        return []
    image_matches = image_matches[image_matches["modality"] == "image"]

    units, seq = [], 0
    for _, m in image_matches.iterrows():
        claim_rows = claims[claims["claim_id"] == m["claim_id"]]
        if claim_rows.empty:
            continue
        claim = claim_rows.iloc[0]
        asset_rows = image_assets[image_assets["asset_id"] == m["asset_id"]]
        asset = asset_rows.iloc[0] if not asset_rows.empty else {}
        product_category_match = bool(asset.get("product_category")) if isinstance(asset, pd.Series) else False
        img_fields = build_image_evidence_unit(claim["claim_category"], m["match_score"], m["match_method"], product_category_match, config)
        seq += 1
        strength = "strong" if m["match_score"] >= 0.4 else "medium"
        units.append({
            "evidence_unit_id": f"eu_img_{seq:04d}", "claim_id": claim["claim_id"], "claim_text": claim["claim_text"],
            "claim_category": claim["claim_category"], "product_name": asset.get("product_name", "") if isinstance(asset, pd.Series) else "",
            "product_category": asset.get("product_category", "") if isinstance(asset, pd.Series) else "",
            "modality": "image", "evidence_id": m["asset_id"], "evidence_text": "",
            "evidence_passage": "", "source_type": "official_product_image",
            "source_url": asset.get("source_url", "") if isinstance(asset, pd.Series) else "", "source_date": "",
            "source_independence": "self_reported", "evidence_strength": strength,
            "relevance_score": m["match_score"], "relevance_method": "clip_vit_b32_text_image_cosine",
            "stance": img_fields["stance"], "stance_score": img_fields["stance_score"],
            "reliability_score": reliability_score(strength, "self_reported", config) if img_fields["stance"] == "supports" else 0.0,
            "visually_groundable": img_fields["visually_groundable"], "visual_gate_passed": img_fields["visual_gate_passed"],
            "visual_gate_reason": img_fields["visual_gate_reason"], "permitted_evidence_role": img_fields["permitted_evidence_role"],
            "brand_evidence_eligible": True, "evaluation_status": "automated_not_human_validated",
            "provisional_finding": True, "external_validation_recommended": False,
            "provenance_note": m["match_explanation"],
        })
    return units


def build_video_units(claims: pd.DataFrame, config: dict) -> list[dict]:
    video_matches = pd.read_csv(PROCESSED / "claim_visual_evidence_matches.csv") if (PROCESSED / "claim_visual_evidence_matches.csv").exists() else pd.DataFrame()
    video_assets = pd.read_csv(INTERIM_DAY2_6 / "video_assets_expanded.csv") if (INTERIM_DAY2_6 / "video_assets_expanded.csv").exists() else pd.DataFrame()
    if video_matches.empty:
        return []
    video_matches = video_matches[video_matches["modality"] == "video"]

    units, seq = [], 0
    for _, m in video_matches.iterrows():
        claim_rows = claims[claims["claim_id"] == m["claim_id"]]
        if claim_rows.empty:
            continue
        claim = claim_rows.iloc[0]
        asset_rows = video_assets[video_assets["video_asset_id"] == m["asset_id"]]
        asset = asset_rows.iloc[0] if not asset_rows.empty else pd.Series({"product_category": "", "transcript_available": False})
        vid_fields = build_video_evidence_unit(
            claim["claim_category"], m["match_score"], m["match_method"], str(asset.get("product_category", "")),
            bool(asset.get("transcript_available", False)), None, config,
        )
        seq += 1
        strength = "strong" if m["match_score"] >= 0.35 else "medium"
        units.append({
            "evidence_unit_id": f"eu_vid_{seq:04d}", "claim_id": claim["claim_id"], "claim_text": claim["claim_text"],
            "claim_category": claim["claim_category"], "product_name": asset.get("product_name", ""),
            "product_category": asset.get("product_category", ""), "modality": "video", "evidence_id": m["asset_id"],
            "evidence_text": "", "evidence_passage": "", "source_type": "official_video",
            "source_url": asset.get("source_url", ""), "source_date": "", "source_independence": "self_reported",
            "evidence_strength": strength, "relevance_score": m["match_score"], "relevance_method": "clip_vit_b32_text_video_cosine",
            "stance": vid_fields["stance"], "stance_score": vid_fields["stance_score"],
            "reliability_score": reliability_score(strength, "self_reported", config) if vid_fields["stance"] == "supports" else 0.0,
            "visually_groundable": vid_fields["visually_groundable"], "visual_gate_passed": vid_fields["visual_gate_passed"],
            "visual_gate_reason": vid_fields["visual_gate_reason"], "permitted_evidence_role": vid_fields["permitted_evidence_role"],
            "brand_evidence_eligible": True, "evaluation_status": "automated_not_human_validated",
            "provisional_finding": True, "external_validation_recommended": False,
            "provenance_note": m["match_explanation"] + f"; shared_gallery={vid_fields['is_shared_gallery_video']}",
        })
    return units


def build_reference_units(claims: pd.DataFrame, config: dict) -> list[dict]:
    ref_matches = pd.read_csv(PROCESSED / "claim_reference_matches.csv") if (PROCESSED / "claim_reference_matches.csv").exists() else pd.DataFrame()
    chunks = pd.read_csv(PROCESSED / "reference_document_chunks.csv") if (PROCESSED / "reference_document_chunks.csv").exists() else pd.DataFrame()
    manifest = pd.read_csv(INTERIM_DAY2_6 / "multimodal_reference_assets.csv") if (INTERIM_DAY2_6 / "multimodal_reference_assets.csv").exists() else pd.DataFrame()
    if ref_matches.empty:
        return []

    units, seq = [], 0
    for _, m in ref_matches.iterrows():
        claim_rows = claims[claims["claim_id"] == m["claim_id"]]
        if claim_rows.empty:
            continue
        claim = claim_rows.iloc[0]
        chunk_rows = chunks[chunks["chunk_id"] == m["chunk_id"]]
        chunk = chunk_rows.iloc[0] if not chunk_rows.empty else pd.Series({"chunk_text": "", "numerical_claims": "", "reference_type": ""})
        ref_rows = manifest[manifest["reference_id"] == m["reference_id"]]
        ref = ref_rows.iloc[0] if not ref_rows.empty else pd.Series({"source_domain": "", "certification_authority": ""})

        ref_fields = classify_reference_evidence(
            claim["claim_text"], chunk.get("chunk_text", ""), chunk.get("numerical_claims", ""),
            chunk.get("reference_type", ""), ref.get("certification_authority", ""), ref.get("source_domain", ""),
            m["match_score"], config,
        )
        seq += 1
        strength = "strong" if m["match_score"] >= 0.5 else "medium"
        units.append({
            "evidence_unit_id": f"eu_ref_{seq:04d}", "claim_id": claim["claim_id"], "claim_text": claim["claim_text"],
            "claim_category": claim["claim_category"], "product_name": "", "product_category": "",
            "modality": "reference", "evidence_id": m["chunk_id"], "evidence_text": chunk.get("chunk_text", ""),
            "evidence_passage": extract_passage(chunk.get("chunk_text", "")),
            "source_type": "certification_registry" if ref_fields["source_independence"] == "independent" else "official_reference",
            "source_url": chunk.get("source_url", ""), "source_date": "",
            "source_independence": ref_fields["source_independence"], "evidence_strength": strength,
            "relevance_score": m["match_score"], "relevance_method": "sentence_transformers_all_MiniLM_L6_v2",
            "stance": ref_fields["stance"], "stance_score": ref_fields["stance_score"],
            "reliability_score": reliability_score(strength, ref_fields["source_independence"], config) if ref_fields["stance"] == "supports" else (0.6 if ref_fields["stance"] == "challenges" else 0.0),
            "visually_groundable": False, "visual_gate_passed": False, "visual_gate_reason": "not applicable to reference documents",
            "permitted_evidence_role": ref_fields["permitted_evidence_role"], "brand_evidence_eligible": True,
            "evaluation_status": "automated_not_human_validated", "provisional_finding": True,
            "external_validation_recommended": ref_fields["reference_category"] == "conflicting_reference_evidence",
            "provenance_note": f"reference_category={ref_fields['reference_category']}; {m['match_explanation']}",
        })
    return units


def build_configuration_units(all_units: pd.DataFrame, modalities: list[str]) -> pd.DataFrame:
    """Restrict evidence units to only the modalities enabled for this
    configuration -- everything else (including the excluded-self-validation
    row) is dropped, so a claim with e.g. text-only config never sees image
    evidence at all."""
    return all_units[all_units["modality"].isin(modalities)].copy()


def main() -> int:
    config = load_config()
    FUSION_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    claims = pd.read_csv(PROCESSED / "claim_multimodal_evidence_candidates.csv")
    if "brand" not in claims.columns:
        claims["brand"] = "TALA"  # this project's fusion universe is TALA-only (verified in Day 2 claim extraction)
    print(f"Loaded {len(claims)} TALA claim bundles (the fusion analytical-unit universe)")

    all_units = []
    seq = 0
    for _, claim in claims.iterrows():
        seq += 1
        all_units.append(build_official_disclosure_unit(claim, 9000 + seq, config))

    print("Building text evidence units ...")
    all_units.extend(build_text_units(claims, config))
    print("Building image evidence units ...")
    all_units.extend(build_image_units(claims, config))
    print("Building video evidence units ...")
    all_units.extend(build_video_units(claims, config))
    print("Building reference evidence units ...")
    all_units.extend(build_reference_units(claims, config))

    units_df = pd.DataFrame(all_units)
    for col in EVIDENCE_UNIT_COLUMNS:
        if col not in units_df.columns:
            units_df[col] = ""
    units_df = units_df[EVIDENCE_UNIT_COLUMNS]
    units_path = FUSION_DIR / "claim_evidence_units.csv"
    units_df.to_csv(units_path, index=False)
    print(f"\nSaved: {units_path.relative_to(PROJECT_ROOT)} ({len(units_df)} evidence units)")
    print(units_df["modality"].value_counts().to_string())

    claims_universe = claims["claim_id"].tolist()

    print("\n[Fusion configurations]")
    config_results = {}
    config_summary_rows = []
    prev_labels = None
    for config_name, modalities in CONFIGURATIONS:
        cfg_units = build_configuration_units(units_df, modalities)
        result = fuse_all_claims(cfg_units, config, claims_universe=claims_universe)
        result["configuration"] = config_name
        config_results[config_name] = result

        label_counts = result["automated_label"].value_counts().to_dict()
        n_changed = 0
        if prev_labels is not None:
            merged = result[["claim_id", "automated_label"]].merge(prev_labels, on="claim_id", suffixes=("", "_prev"))
            n_changed = int((merged["automated_label"] != merged["automated_label_prev"]).sum())
        prev_labels = result[["claim_id", "automated_label"]].rename(columns={"automated_label": "automated_label"})

        config_summary_rows.append({
            "configuration": config_name, "modalities": ";".join(modalities),
            "claims_processed": len(result),
            "claims_with_sufficient_evidence": int((result["automated_label"] != "insufficient_evidence").sum()),
            "aligned": label_counts.get("aligned", 0), "partially_aligned": label_counts.get("partially_aligned", 0),
            "mixed": label_counts.get("mixed", 0), "divergent": label_counts.get("divergent", 0),
            "insufficient_evidence": label_counts.get("insufficient_evidence", 0),
            "mean_confidence": round(float(result["confidence"].mean()), 4),
            "evidence_coverage_pct": round(100 * (result["evidence_count"] > 0).mean(), 2),
            "claims_changed_from_previous": n_changed,
        })
        print(f"  [{config_name}] " + ", ".join(f"{k}={v}" for k, v in label_counts.items()))

    all_config_df = pd.concat(config_results.values(), ignore_index=True)
    all_config_df.to_csv(FUSION_DIR / "fusion_configuration_results.csv", index=False)

    final_results = config_results["text_image_video_reference"].drop(columns=["configuration"])
    final_results.to_csv(FUSION_DIR / "claim_fusion_results.csv", index=False)
    print(f"\nSaved: {(FUSION_DIR / 'claim_fusion_results.csv').relative_to(PROJECT_ROOT)} ({len(final_results)} claims)")

    config_comparison = pd.DataFrame(config_summary_rows)
    config_comparison.to_csv(TABLES / "fusion_configuration_comparison.csv", index=False)
    print(f"Saved: {(TABLES / 'fusion_configuration_comparison.csv').relative_to(PROJECT_ROOT)}")

    # ── Modality contribution: eligible-evidence-unit counts by modality ───────
    modality_contribution = units_df.groupby("modality").agg(
        n_units=("evidence_unit_id", "count"),
        n_supports=("stance", lambda s: (s == "supports").sum()),
        n_challenges=("stance", lambda s: (s == "challenges").sum()),
        n_context_only=("stance", lambda s: (s == "neutral_context").sum()),
        n_gate_passed=("visual_gate_passed", "sum"),
    ).reset_index()
    modality_contribution.to_csv(TABLES / "fusion_modality_contribution.csv", index=False)
    print(f"Saved: {(TABLES / 'fusion_modality_contribution.csv').relative_to(PROJECT_ROOT)}")

    # ── Label distribution (final configuration) ────────────────────────────────
    label_dist = final_results["automated_label"].value_counts().reset_index()
    label_dist.columns = ["automated_label", "n_claims"]
    label_dist["business_facing_label"] = label_dist["automated_label"].map(
        {"aligned": "automated alignment signal", "partially_aligned": "automated partial-alignment signal",
         "mixed": "automated mixed-evidence signal", "divergent": "potential divergence",
         "insufficient_evidence": "insufficient evidence"})
    label_dist.to_csv(TABLES / "fusion_label_distribution.csv", index=False)
    print(f"Saved: {(TABLES / 'fusion_label_distribution.csv').relative_to(PROJECT_ROOT)}")

    # ── Divergence matrix: category x label ─────────────────────────────────────
    divergence_matrix = final_results.pivot_table(
        index="claim_category", columns="automated_label", values="claim_id", aggfunc="count", fill_value=0)
    divergence_matrix.to_csv(TABLES / "claim_divergence_matrix.csv")
    print(f"Saved: {(TABLES / 'claim_divergence_matrix.csv').relative_to(PROJECT_ROOT)}")

    # ── Evidence gaps ────────────────────────────────────────────────────────────
    gaps = final_results[final_results["automated_label"] == "insufficient_evidence"][
        ["claim_id", "claim_category", "missing_modalities", "evidence_count"]]
    gaps.to_csv(TABLES / "fusion_evidence_gaps.csv", index=False)
    print(f"Saved: {(TABLES / 'fusion_evidence_gaps.csv').relative_to(PROJECT_ROOT)}")

    # ── Review flags ─────────────────────────────────────────────────────────────
    flags = final_results.copy()
    flags["sensitive_to_weights"] = ""  # filled in by scripts/run_fusion_sensitivity.py
    flags["single_evidence_dependency"] = flags["eligible_evidence_count"] == 1
    flags["conflicting_evidence"] = flags["evidence_conflict"]
    flags_out = flags[[
        "claim_id", "automated_label", "business_facing_label", "confidence", "self_reported_only",
        "single_evidence_dependency", "conflicting_evidence", "missing_modalities",
        "external_validation_recommended", "presentation_restriction",
    ]].rename(columns={"confidence": "automated_confidence"})
    flags_out["flag_reason"] = flags_out.apply(
        lambda r: "; ".join(filter(None, [
            "self_reported_only" if r["self_reported_only"] else "",
            "single_evidence_dependency" if r["single_evidence_dependency"] else "",
            "conflicting_evidence" if r["conflicting_evidence"] else "",
            "missing_modalities" if r["missing_modalities"] else "",
        ])), axis=1)
    flags_out.to_csv(TABLES / "fusion_review_flags.csv", index=False)
    print(f"Saved: {(TABLES / 'fusion_review_flags.csv').relative_to(PROJECT_ROOT)}")

    # ── Figures ──────────────────────────────────────────────────────────────────
    n_claims_total = len(final_results)
    caption = f"n={n_claims_total} claims -- automated, not human validated -- see docs/automated_fusion_evaluation.md for limitations"

    fig, ax = plt.subplots(figsize=(8, 5))
    label_dist.set_index("automated_label")["n_claims"].plot(kind="bar", ax=ax, color="#4C72B0")
    ax.set_title("Fusion label distribution (automated, not human-validated)")
    ax.set_xlabel("Automated label"); ax.set_ylabel("Number of claims")
    fig.text(0.5, -0.05, caption, ha="center", fontsize=8, wrap=True)
    fig.tight_layout()
    fig.savefig(FIGURES / "fusion_label_distribution.png", bbox_inches="tight", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    modality_contribution.set_index("modality")[["n_units"]].plot(kind="bar", ax=ax, color="#55A868", legend=False)
    ax.set_title("Evidence unit coverage by modality")
    ax.set_xlabel("Modality"); ax.set_ylabel("Evidence units")
    fig.text(0.5, -0.05, caption, ha="center", fontsize=8, wrap=True)
    fig.tight_layout()
    fig.savefig(FIGURES / "fusion_modality_coverage.png", bbox_inches="tight", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    config_comparison.set_index("configuration")[["aligned", "partially_aligned", "mixed", "divergent", "insufficient_evidence"]].plot(
        kind="bar", stacked=True, ax=ax)
    ax.set_title("Label distribution across fusion configurations (automated, not human-validated)")
    ax.set_xlabel("Configuration"); ax.set_ylabel("Number of claims")
    fig.text(0.5, -0.15, caption, ha="center", fontsize=8, wrap=True)
    fig.tight_layout()
    fig.savefig(FIGURES / "fusion_configuration_comparison.png", bbox_inches="tight", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(divergence_matrix.values, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(divergence_matrix.columns))); ax.set_xticklabels(divergence_matrix.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(divergence_matrix.index))); ax.set_yticklabels(divergence_matrix.index)
    for i in range(divergence_matrix.shape[0]):
        for j in range(divergence_matrix.shape[1]):
            ax.text(j, i, int(divergence_matrix.values[i, j]), ha="center", va="center")
    ax.set_title("Claim category x automated label (heatmap)")
    plt.colorbar(im, ax=ax)
    fig.text(0.5, -0.05, caption, ha="center", fontsize=8, wrap=True)
    fig.tight_layout()
    fig.savefig(FIGURES / "claim_divergence_heatmap.png", bbox_inches="tight", dpi=150)
    plt.close(fig)

    print(f"\nSaved 4 figures to {FIGURES.relative_to(PROJECT_ROOT)}")
    print("\n" + "=" * 60)
    print("Day 3A primary multimodal fusion complete.")
    print(f"  Claims processed: {len(final_results)}")
    print(f"  Final label distribution: {final_results['automated_label'].value_counts().to_dict()}")
    print("  All results: evaluation_status=automated_not_human_validated, provisional_finding=True")
    return 0


if __name__ == "__main__":
    sys.exit(main())
