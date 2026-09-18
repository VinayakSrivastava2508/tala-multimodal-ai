"""Recovery script: rebuild RAG corpora (Part G) + gates/audit (Parts H-I) from
already-saved Day 2 interim files, with no new network requests.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.hydrate_day2_sources import (
    build_rag_corpora,
    check_quality_gates,
    generate_audit_outputs,
    INTERIM_DAY2,
    CORPORA_DIR,
    TABLES,
    NOW_ISO,
)

TABLES.mkdir(parents=True, exist_ok=True)
CORPORA_DIR.mkdir(parents=True, exist_ok=True)

print("Loading saved Day 2 interim files...")
creators_enriched  = pd.read_csv(INTERIM_DAY2 / "creator_posts_enriched.csv")
press_hydrated      = pd.read_csv(INTERIM_DAY2 / "hydrated_press.csv")
claims_hydrated      = pd.read_csv(INTERIM_DAY2 / "hydrated_claims.csv")
reviews_labeled      = pd.read_csv(INTERIM_DAY2 / "reviews_labeled.csv")
platforms_enriched  = pd.read_csv(INTERIM_DAY2 / "platform_strategy_enriched.csv")
exp_path = INTERIM_DAY2 / "claims_expanded.csv"
claims_expanded      = pd.read_csv(exp_path) if exp_path.exists() else pd.DataFrame()

# claims_expanded.csv predates the source_id column; recover it from claim_id
# (format "c_<source_id>_<NN>") so page-level evidence strength can be joined in.
if not claims_expanded.empty and "source_id" not in claims_expanded.columns:
    claims_expanded["source_id"] = claims_expanded["claim_id"].str.extract(r"^c_(.+)_\d{2}$")

print(f"  claims_expanded={len(claims_expanded)}, claims_hydrated={len(claims_hydrated)}, "
      f"reviews={len(reviews_labeled)}, press={len(press_hydrated)}, "
      f"creators={len(creators_enriched)}, platforms={len(platforms_enriched)}")

# ── Part G: rebuild RAG corpora with source-page evidence strength for claims ──
print("\n[PART G] Rebuilding RAG corpora (official_claims strength now from source page)...")
claims_corpus, ce_corpus, cs_corpus = build_rag_corpora(
    claims_expanded, reviews_labeled, press_hydrated,
    creators_enriched, platforms_enriched,
    claims_hydrated=claims_hydrated,
)
claims_corpus.to_csv(CORPORA_DIR / "official_claims_corpus.csv", index=False)
ce_corpus.to_csv(CORPORA_DIR / "customer_experience_corpus.csv", index=False)
cs_corpus.to_csv(CORPORA_DIR / "creator_strategy_corpus.csv", index=False)
print(f"  official_claims_corpus: {len(claims_corpus)} "
      f"(strong={int((claims_corpus['evidence_strength']=='strong').sum())})")

# ── Part H ─────────────────────────────────────────────────────────────────────
print("\n[PART H] Checking quality gates...")
gates_df = check_quality_gates(claims_corpus, ce_corpus, cs_corpus, creators_enriched)
print(gates_df[["gate", "threshold", "actual", "status"]].to_string(index=False))
print(f"\n  Overall: {gates_df['overall'].iloc[0]}")
gates_df.to_csv(TABLES / "day2_quality_gate_report.csv", index=False)
print("  Saved: outputs/tables/day2_quality_gate_report.csv")

# ── Part I ─────────────────────────────────────────────────────────────────────
print("\n[PART I] Generating audit outputs...")
generate_audit_outputs(
    press_hydrated, claims_hydrated, claims_expanded, reviews_labeled,
    creators_enriched, platforms_enriched,
    claims_corpus, ce_corpus, cs_corpus, gates_df,
)

# ── hydrated_sources.csv (combined) ───────────────────────────────────────────
frames = []
for src_name, df, strength_col, status_col in [
    ("press",     press_hydrated,     "h_evidence_strength", "h_hydration_status"),
    ("claims",    claims_hydrated,    "h_evidence_strength", "h_hydration_status"),
    ("reviews",   reviews_labeled,    "h_evidence_strength", "h_hydration_status"),
    ("platforms", platforms_enriched, "evidence_strength",   "h_hydration_status"),
]:
    if df.empty:
        continue
    tmp = pd.DataFrame({
        "source_type":       src_name,
        "source_id":         df.get("source_id", pd.Series(range(len(df)))),
        "brand":             df.get("brand", ""),
        "source_url":        df.get("source_url", ""),
        "hydration_status":  df.get(status_col, "unknown"),
        "evidence_strength": df.get(strength_col, "unusable"),
        "rag_usable":        df.get("h_rag_usable", False),
        "text_length":       df.get("h_text_length", 0),
        "retrieved_at":      NOW_ISO,
    })
    frames.append(tmp)

if frames:
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(INTERIM_DAY2 / "hydrated_sources.csv", index=False)
    print(f"  Saved: data/interim/day2/hydrated_sources.csv ({len(combined)} rows)")

print("\nDay 2 Parts G-I complete.")
sys.exit(0 if gates_df["overall"].iloc[0] == "GO" else 1)
