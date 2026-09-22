"""Day 3B: transparent, deterministic reciprocal-rank fusion (RRF) combining
candidate lists retrieved independently from multiple Chroma collections.
NLI is never used as a relevance/stance authority here -- only vector
similarity, direct claim linkage, evidence strength, and source independence,
all of which are logged in the returned trace."""

from __future__ import annotations

from typing import Dict, List, Optional

RRF_K = 60  # standard RRF smoothing constant

EVIDENCE_STRENGTH_WEIGHT = {"strong": 1.0, "medium": 0.7, "weak": 0.4, "unusable": 0.0}
SOURCE_INDEPENDENCE_WEIGHT = {"independent": 1.0, "customer_reported": 0.85, "self_reported": 0.5, "unclear": 0.4}
DIRECT_LINK_BONUS = 1.0  # added on top of RRF score for direct claim-evidence links


def _evidence_strength_score(metadata: dict) -> float:
    return EVIDENCE_STRENGTH_WEIGHT.get(metadata.get("evidence_strength", ""), 0.5)


def _source_independence_score(metadata: dict) -> float:
    raw = str(metadata.get("source_independence", ""))
    tokens = [t for t in raw.split(";") if t]
    if not tokens:
        return SOURCE_INDEPENDENCE_WEIGHT.get("unclear", 0.4)
    return max(SOURCE_INDEPENDENCE_WEIGHT.get(t, 0.4) for t in tokens)


def fuse_candidates(
    candidate_lists: Dict[str, List[dict]],
    max_results: int = 10,
    source_diversity_cap: Optional[int] = None,
) -> List[dict]:
    """candidate_lists: {list_name: [candidate, ...]} where each candidate has
    id/collection/metadata/distance/similarity/original_rank(/direct_link).
    Returns a deduplicated, ranked list; every item's 'trace' explains its
    final ranking and, for ranked-out items, its exclusion_reason."""
    by_id: Dict[str, dict] = {}

    for list_name, candidates in candidate_lists.items():
        for c in candidates:
            cid = c["id"]
            rrf_contribution = 1.0 / (RRF_K + c["original_rank"] + 1)
            if cid not in by_id:
                by_id[cid] = {
                    "id": cid, "collection": c["collection"], "metadata": c["metadata"],
                    "document": c.get("document"), "best_similarity": c["similarity"],
                    "rrf_score": 0.0, "direct_link": bool(c.get("direct_link", False)),
                    "trace": {"sources": [], "rank_fusion_contribution": 0.0},
                }
            entry = by_id[cid]
            entry["best_similarity"] = max(entry["best_similarity"], c["similarity"])
            entry["rrf_score"] += rrf_contribution
            entry["direct_link"] = entry["direct_link"] or bool(c.get("direct_link", False))
            entry["trace"]["sources"].append({
                "list": list_name, "collection": c["collection"], "original_rank": c["original_rank"],
                "distance": c["distance"], "similarity": c["similarity"], "rrf_contribution": round(rrf_contribution, 6),
            })

    scored = []
    for cid, entry in by_id.items():
        metadata = entry["metadata"]
        strength_score = _evidence_strength_score(metadata)
        independence_score = _source_independence_score(metadata)
        direct_link_bonus = DIRECT_LINK_BONUS if entry["direct_link"] else 0.0
        final_score = entry["rrf_score"] + direct_link_bonus + 0.1 * strength_score + 0.1 * independence_score
        entry["trace"]["rank_fusion_contribution"] = round(entry["rrf_score"], 6)
        entry["trace"]["direct_link_bonus"] = direct_link_bonus
        entry["trace"]["evidence_strength_score"] = strength_score
        entry["trace"]["source_independence_score"] = independence_score
        entry["final_score"] = round(final_score, 6)
        scored.append(entry)

    scored.sort(key=lambda e: e["final_score"], reverse=True)

    # Source diversity: cap how many results may come from the same source_type
    # so one dominant document type can't crowd out the rest.
    kept = []
    excluded = []
    source_type_counts: Dict[str, int] = {}
    for entry in scored:
        source_type = entry["metadata"].get("source_type", "unknown")
        count = source_type_counts.get(source_type, 0)
        if source_diversity_cap and count >= source_diversity_cap and len(kept) < max_results:
            entry["trace"]["exclusion_reason"] = f"source_diversity_cap_exceeded_for_{source_type}"
            excluded.append(entry)
            continue
        if len(kept) >= max_results:
            entry["trace"]["exclusion_reason"] = "max_results_exceeded"
            excluded.append(entry)
            continue
        source_type_counts[source_type] = count + 1
        entry["trace"]["exclusion_reason"] = None
        kept.append(entry)

    for rank, entry in enumerate(kept):
        entry["final_rank"] = rank

    return kept + excluded
