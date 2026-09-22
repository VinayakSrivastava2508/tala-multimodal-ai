"""Day 3B: idempotent builder for the three persistent ChromaDB collections
(tala_claims, tala_text_evidence, tala_visual_evidence) at data/vector_db/chroma/.

Reruns upsert-by-deterministic-ID (never duplicate) and remove any record
whose ID is no longer present in the authoritative Day 1-3A corpus.

Usage: python scripts/build_chroma_index.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag import chroma_store, corpus_builder, embedders  # noqa: E402
from src.rag.schemas import (  # noqa: E402
    CHROMA_PERSIST_DIR, CLIP_EMBEDDING_DIM, COLLECTION_CLAIMS, COLLECTION_TEXT_EVIDENCE,
    COLLECTION_VISUAL_EVIDENCE, TEXT_EMBEDDING_DIM, sha256_text,
)

TABLES = PROJECT_ROOT / "outputs" / "tables"
NOW_ISO = datetime.now(timezone.utc).isoformat()


def _build_and_upsert_text_collection(client, name: str, records: list[dict]) -> dict:
    collection = chroma_store.get_or_create_collection(client, name)
    seen_ids = set()
    deduped = []
    for r in records:
        if r["id"] in seen_ids:
            continue
        seen_ids.add(r["id"])
        deduped.append(r)

    texts = [r["embedding_text"] for r in deduped]
    embeddings = embedders.embed_texts(texts)
    for emb in embeddings:
        chroma_store.validate_embedding_dimension(emb, "text")

    n_upserted = chroma_store.upsert_records(collection, deduped, embeddings)
    n_removed = chroma_store.remove_stale_records(collection, seen_ids)
    return {"collection": name, "n_upserted": n_upserted, "n_removed_stale": n_removed, "n_final": collection.count(), "records": deduped}


def _build_and_upsert_visual_collection(client, name: str, records: list[dict]) -> dict:
    collection = chroma_store.get_or_create_collection(client, name)
    seen_ids = set()
    embeddable = []
    n_missing_asset = 0
    n_embed_failed = 0

    for r in records:
        if r["id"] in seen_ids:
            continue
        local_path = Path(r["local_path"])
        if not local_path.exists():
            n_missing_asset += 1
            continue
        emb = embedders.embed_image(str(local_path))
        if emb is None:
            n_embed_failed += 1
            continue
        chroma_store.validate_embedding_dimension(emb, r["modality"])
        seen_ids.add(r["id"])
        embeddable.append((r, emb))

    if embeddable:
        recs = [r for r, _ in embeddable]
        embs = [e for _, e in embeddable]
        n_upserted = chroma_store.upsert_records(collection, recs, embs)
    else:
        n_upserted = 0
    n_removed = chroma_store.remove_stale_records(collection, seen_ids)
    return {
        "collection": name, "n_upserted": n_upserted, "n_removed_stale": n_removed,
        "n_final": collection.count(), "n_missing_asset": n_missing_asset, "n_embed_failed": n_embed_failed,
        "records": [r for r, _ in embeddable],
    }


def _write_manifest(build_results: dict) -> None:
    manifest = {"generated_at_utc": NOW_ISO, "chroma_persist_dir": str(CHROMA_PERSIST_DIR.relative_to(PROJECT_ROOT)), "collections": {}}
    for name, result in build_results.items():
        entries = []
        for r in result["records"]:
            content = r.get("document") or json.dumps(r["metadata"], sort_keys=True)
            entries.append({"id": r["id"], "content_sha256": sha256_text(content)})
        manifest["collections"][name] = {
            "n_final": result["n_final"], "n_upserted": result["n_upserted"],
            "n_removed_stale": result["n_removed_stale"], "entries": entries,
        }
    manifest_path = PROJECT_ROOT / "data" / "vector_db" / "chroma_index_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved: {manifest_path.relative_to(PROJECT_ROOT)}")


def _write_summary_tables(build_results: dict) -> None:
    summary_rows = []
    for name, result in build_results.items():
        summary_rows.append({
            "collection": name, "n_final": result["n_final"], "n_upserted_this_run": result["n_upserted"],
            "n_removed_stale_this_run": result["n_removed_stale"],
            "n_missing_asset_rejected": result.get("n_missing_asset", 0),
            "n_embed_failed_rejected": result.get("n_embed_failed", 0),
        })
    pd.DataFrame(summary_rows).to_csv(TABLES / "rag_collection_summary.csv", index=False)
    print(f"Saved: {(TABLES / 'rag_collection_summary.csv').relative_to(PROJECT_ROOT)}")

    coverage_rows = []
    for name, result in build_results.items():
        df = pd.DataFrame([r["metadata"] for r in result["records"]])
        if df.empty:
            continue
        if "brand" in df.columns:
            for (brand,), grp in df.groupby(["brand"]):
                coverage_rows.append({"collection": name, "dimension": "brand", "value": brand, "n_rows": len(grp)})
        if "source_type" in df.columns:
            for (st,), grp in df.groupby(["source_type"]):
                coverage_rows.append({"collection": name, "dimension": "source_type", "value": st, "n_rows": len(grp)})
        if "evidence_strength" in df.columns:
            for (es,), grp in df.groupby(["evidence_strength"]):
                coverage_rows.append({"collection": name, "dimension": "evidence_strength", "value": es, "n_rows": len(grp)})
    pd.DataFrame(coverage_rows).to_csv(TABLES / "rag_corpus_coverage.csv", index=False)
    print(f"Saved: {(TABLES / 'rag_corpus_coverage.csv').relative_to(PROJECT_ROOT)}")

    modality_rows = []
    for name, result in build_results.items():
        df = pd.DataFrame([r["metadata"] for r in result["records"]])
        if df.empty or "modality" not in df.columns:
            continue
        for (modality,), grp in df.groupby(["modality"]):
            modality_rows.append({"collection": name, "modality": modality, "n_rows": len(grp)})
    pd.DataFrame(modality_rows).to_csv(TABLES / "rag_modality_coverage.csv", index=False)
    print(f"Saved: {(TABLES / 'rag_modality_coverage.csv').relative_to(PROJECT_ROOT)}")


def main() -> int:
    print(f"Building ChromaDB index at {CHROMA_PERSIST_DIR.relative_to(PROJECT_ROOT)} ...")
    client = chroma_store.get_client()

    claims_records = corpus_builder.build_claims_records()
    text_records = corpus_builder.build_text_evidence_records()
    visual_records = corpus_builder.build_visual_evidence_records()

    print(f"  Loaded {len(claims_records)} claim records, {len(text_records)} text-evidence records, "
          f"{len(visual_records)} visual-evidence candidate records")

    build_results = {}
    build_results[COLLECTION_CLAIMS] = _build_and_upsert_text_collection(client, COLLECTION_CLAIMS, claims_records)
    build_results[COLLECTION_TEXT_EVIDENCE] = _build_and_upsert_text_collection(client, COLLECTION_TEXT_EVIDENCE, text_records)
    build_results[COLLECTION_VISUAL_EVIDENCE] = _build_and_upsert_visual_collection(client, COLLECTION_VISUAL_EVIDENCE, visual_records)

    for name, result in build_results.items():
        print(f"  [{name}] final={result['n_final']} upserted={result['n_upserted']} "
              f"removed_stale={result['n_removed_stale']}"
              + (f" missing_asset_rejected={result.get('n_missing_asset', 0)} embed_failed={result.get('n_embed_failed', 0)}"
                 if "n_missing_asset" in result else ""))

    _write_manifest(build_results)
    _write_summary_tables(build_results)

    print("\nDay 3B ChromaDB index build complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
