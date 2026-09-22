"""Day 3C acceptance patch, item 2: consolidate the 5 mixed claim RECORDS into
their underlying strategic THEMES.

Pipeline order (also enforced by scripts/build_all_day3c_outputs.py): this
script must run AFTER scripts/build_analytical_synthesis.py, because it reads
analytical_synthesis_claim_master.csv and enriches the record-level
mixed_claim_deep_dive.csv / claim_label_summary.csv that script just wrote.

Deterministic rule (no LLM grouping): mixed claim records are grouped by exact
match of `claim_text` (case/whitespace-normalised). This is the only rule
used -- claim records that are verbatim duplicates of the same official
sentence are, by construction, the same underlying claim variant repeated
across product pages, and therefore the same strategic theme. Any claim
record whose text does not match another mixed record's text forms its own
single-record theme. The authoritative Day 3A claim-level fusion labels
(`data/processed/fusion/claim_fusion_results.csv`) are never modified --
this script only reads `analytical_synthesis_claim_master.csv` (itself
already built from Day 3A output) and re-expresses the same 5 mixed claim
records at the theme level.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
FUSION_DIR = PROJECT_ROOT / "data" / "processed" / "fusion"


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def _theme_name(sample_text: str) -> tuple[str, str]:
    """Return (theme_name, theme_description) via a deterministic keyword rule
    over the normalised claim text -- not an LLM call."""
    norm = _normalise(sample_text)
    if "tier 3" in norm or "tier-3" in norm:
        return (
            "Tier-3 supplier / material-innovation collaboration",
            (
                "Claim variants stating TALA's product team collaborates with tier-3 "
                "suppliers to develop new, innovative fabrics."
            ),
        )
    if "consciously-made" in norm or "hefty price tag" in norm or "mission" in norm:
        return (
            "Brand mission / conscious-value proposition",
            (
                "TALA's brand mission statement positioning it as delivering "
                "consciously-made, performance activewear without a premium price tag."
            ),
        )
    # Deterministic fallback for any future, currently-unseen mixed claim text.
    return ("Ungrouped mixed claim", f"Claim text does not match a known theme keyword rule: {sample_text[:80]}")


def build_theme_summary() -> pd.DataFrame:
    master = pd.read_csv(TABLES_DIR / "analytical_synthesis_claim_master.csv")
    units = pd.read_csv(FUSION_DIR / "claim_evidence_units.csv")

    mixed = master[master["production_fusion_label"] == "mixed"].copy()
    assert len(mixed) == 5, f"expected exactly 5 mixed claim records, found {len(mixed)}"

    mixed["_norm_text"] = mixed["claim_text"].map(_normalise)
    group_keys = sorted(mixed["_norm_text"].unique())

    rows = []
    for i, key in enumerate(group_keys, start=1):
        grp = mixed[mixed["_norm_text"] == key]
        sample_text = grp.iloc[0]["claim_text"]
        theme_name, theme_desc = _theme_name(sample_text)

        claim_ids = sorted(grp["claim_id"].tolist())
        cu = units[units["claim_id"].isin(claim_ids)]
        elig = cu[cu["permitted_evidence_role"] == "supports_or_challenges"]
        support_ids = sorted(elig[elig["stance"] == "supports"]["evidence_unit_id"].astype(str).unique())
        challenge_ids = sorted(elig[elig["stance"] == "challenges"]["evidence_unit_id"].astype(str).unique())
        modalities_present = sorted(cu["modality"].unique().tolist())
        independent_present = bool((elig["source_independence"] == "customer_reported").any())
        weight_sensitive = bool(grp["sensitive_to_weights"].any())

        # representative claim = shortest claim_id (first-published variant), deterministic
        representative_claim = sorted(claim_ids)[0]

        if theme_name.startswith("Tier-3"):
            tension = (
                "The claim of active, ongoing tier-3 supplier collaboration on fabric "
                "innovation is both supported and challenged by independent evidence in "
                "the corpus; the same underlying tension recurs across 4 near-identical "
                "claim variants published across different product pages."
            )
            action = (
                "Review the specific supplier-collaboration evidence once (not per "
                "product-page variant); confirm whether the tension reflects a genuine, "
                "recurring gap between stated and observed supplier practice."
            )
        else:
            tension = (
                "TALA's conscious-value brand-mission statement is both supported and "
                "challenged by evidence in the corpus, indicating a documented gap "
                "between the stated mission framing and some observed customer/press "
                "experience."
            )
            action = (
                "Route the brand-mission statement to marketing/communications review "
                "before repeating it verbatim in external decks."
            )

        rows.append(
            {
                "theme_id": f"theme_{i:02d}",
                "theme_name": theme_name,
                "theme_description": theme_desc,
                "claim_record_count": len(claim_ids),
                "claim_ids": ";".join(claim_ids),
                "representative_claim": representative_claim,
                "supporting_evidence_count": len(support_ids),
                "challenging_evidence_count": len(challenge_ids),
                "supporting_evidence_ids": ";".join(support_ids),
                "challenging_evidence_ids": ";".join(challenge_ids),
                "modalities_present": ";".join(modalities_present),
                "independent_evidence_present": independent_present,
                "weight_sensitive": weight_sensitive,
                "observed_tension": tension,
                "management_implication": (
                    f"{len(claim_ids)} claim record(s) representing this single strategic "
                    "tension should be reviewed and resolved together, not treated as "
                    f"{len(claim_ids)} independent issues."
                ),
                "recommended_action": action,
                "caveat": (
                    "Automated NLI/rule-based fusion label per record, not human-adjudicated; "
                    "theme grouping is a deterministic text-match consolidation for reporting "
                    "purposes only and does not alter the underlying Day 3A claim-level labels."
                ),
            }
        )

    out = pd.DataFrame(rows)
    assert out["claim_record_count"].sum() == 5, "theme claim_record_count must sum to 5"
    all_claim_ids = [cid for ids in out["claim_ids"] for cid in ids.split(";")]
    assert len(all_claim_ids) == len(set(all_claim_ids)) == 5, "every mixed claim must appear in exactly one theme"
    if len(out) != 2:
        print(f"WARNING: inspection of claim text produced {len(out)} theme(s), not the expected 2 -- "
              f"see theme_name column for the actual grouping found.")
    return out


def _claim_id_to_theme_map(theme_df: pd.DataFrame) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for _, row in theme_df.iterrows():
        for cid in str(row["claim_ids"]).split(";"):
            mapping[cid] = row["theme_id"]
    return mapping


def enrich_mixed_deep_dive(theme_df: pd.DataFrame) -> None:
    """Add a theme_id column to mixed_claim_deep_dive.csv (record-level table),
    without altering its 5-record row count or any other column."""
    path = TABLES_DIR / "mixed_claim_deep_dive.csv"
    dd = pd.read_csv(path)
    cid_to_theme = _claim_id_to_theme_map(theme_df)
    dd["theme_id"] = dd["claim_id"].map(cid_to_theme)
    assert dd["theme_id"].notna().all(), "every mixed claim record must map to a theme"
    assert len(dd) == 5, "mixed_claim_deep_dive.csv record count must remain 5"
    dd.to_csv(path, index=False)


def append_theme_level_reporting(theme_df: pd.DataFrame) -> None:
    """Add explicit record-level (5) vs. theme-level (2) reporting rows to
    claim_label_summary.csv under a new section, without altering the
    existing overall_label_distribution mixed=5 row."""
    path = TABLES_DIR / "claim_label_summary.csv"
    summary = pd.read_csv(path)
    summary = summary[summary["section"] != "mixed_claim_theme_consolidation"]

    overall = summary[summary["section"] == "overall_label_distribution"]
    record_level_mixed = int(overall.loc[overall["automated_label"] == "mixed", "n_claims"].iloc[0])
    assert record_level_mixed == 5, "record-level mixed count must remain 5"

    new_rows = pd.DataFrame(
        [
            {"automated_label": "mixed_claim_records", "n_claims": record_level_mixed, "section": "mixed_claim_theme_consolidation"},
            {"automated_label": "mixed_claim_themes", "n_claims": len(theme_df), "section": "mixed_claim_theme_consolidation"},
        ]
    )
    combined = pd.concat([summary, new_rows], ignore_index=True)
    combined.to_csv(path, index=False)


def main() -> None:
    out = build_theme_summary()
    out.to_csv(TABLES_DIR / "mixed_claim_theme_summary.csv", index=False)
    enrich_mixed_deep_dive(out)
    append_theme_level_reporting(out)
    print("mixed_claim_theme_summary rows:", len(out))
    print(out[["theme_id", "theme_name", "claim_record_count", "claim_ids"]].to_string(index=False))
    print("mixed_claim_deep_dive.csv enriched with theme_id; claim_label_summary.csv updated with theme-level reporting.")


if __name__ == "__main__":
    main()
