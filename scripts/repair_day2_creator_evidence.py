"""Day 2 Task 1B: repair and revalidate creator evidence.

Repairs data/interim/day2/creator_posts_enriched.csv so every row marked
usable_as_creator_post=True has a direct social post/video URL, an identifiable
creator, a verified brand link, an evidence-strength classification, and valid
provenance. Invalid rows are downgraded (usable_as_creator_post=False), never
deleted -- they're retained as discovery leads.

If fewer than 20 verified strong/medium creator posts remain after repair, runs
targeted replenishment (real DDG/YouTube-oEmbed discovery, same verification
rules -- no fabricated rows, no lowered thresholds).

Usage: python scripts/repair_day2_creator_evidence.py
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.collectors.creator_identity import (
    BRAND_ALIASES,
    classify_partnership,
    classify_url,
    compute_creator_evidence_strength,
    extract_tiktok_handle,
    resolve_instagram_identity,
    resolve_youtube_identity,
    verify_brand_link,
)

INTERIM_DAY2 = PROJECT_ROOT / "data" / "interim" / "day2"
ARCHIVE_DIR = INTERIM_DAY2 / "archive"
ENRICHED_PATH = INTERIM_DAY2 / "creator_posts_enriched.csv"
ARCHIVE_PATH = ARCHIVE_DIR / "creator_posts_enriched_pre_repair.csv"
NOW_ISO = datetime.now(timezone.utc).isoformat()
TODAY = datetime.now(timezone.utc).strftime("%Y-%m-%d")
VERIFIED_TARGET = 20
BRANDS = ["TALA", "Adanola", "Girlfriend Collective", "Oner Active"]

REPAIR_COLUMNS = [
    "url_class", "direct_post_url", "direct_post_id", "url_validation_status", "invalidation_reason",
    "creator_handle", "creator_name", "creator_identity_source", "creator_identity_confidence",
    "creator_identity_evidence", "creator_identity_verified_at",
    "brand_link_verified", "brand_link_evidence", "brand_link_source", "brand_link_confidence",
    "partnership_type", "partnership_evidence", "partnership_confidence", "partnership_inferred",
    "evidence_strength", "usable_as_creator_post", "source_record_ids",
]


def _effective_url(row: pd.Series) -> str:
    for col in ("post_url", "source_url"):
        val = row.get(col)
        if pd.notna(val) and str(val).strip():
            return str(val)
    return ""


def _nz(val) -> str:
    """Null-safe string coercion: NaN/None -> ''."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    s = str(val)
    return "" if s.lower() == "nan" else s


def repair_row(row: pd.Series, yt_get=None, yt_cache_dir=None) -> Dict:
    """Apply Parts B-F to a single creator_posts_enriched row. Returns the dict of
    new/updated fields."""
    url = _effective_url(row)
    urlinfo = classify_url(url)

    brand = _nz(row.get("brand"))
    platform = str(row.get("platform", "")).lower()
    channel_name = _nz(row.get("channel_name"))
    creator_name_existing = _nz(row.get("creator_name"))
    video_title = _nz(row.get("video_title"))
    caption = _nz(row.get("caption_or_description"))
    personalised_code = _nz(row.get("personalised_code"))
    disclosure_type = _nz(row.get("disclosure_type"))
    existing_handle = _nz(row.get("creator_handle"))

    identity = {
        "creator_handle": "", "creator_name": "", "creator_identity_source": "none",
        "creator_identity_confidence": "unresolved", "creator_identity_evidence": "",
    }
    if urlinfo["url_validation_status"] == "valid":
        if urlinfo["url_class"] == "tiktok_video":
            handle = extract_tiktok_handle(urlinfo["direct_post_url"])
            if handle:
                identity = {
                    "creator_handle": handle, "creator_name": handle,
                    "creator_identity_source": "tiktok_url", "creator_identity_confidence": "high",
                    "creator_identity_evidence": f"handle parsed directly from URL path: @{handle}",
                }
        elif urlinfo["url_class"] in ("youtube_video", "youtube_short"):
            kwargs = dict(fallback_channel_name=channel_name)
            if yt_get is not None:
                kwargs["get"] = yt_get
            if yt_cache_dir is not None:
                kwargs["cache_dir"] = yt_cache_dir
            identity = resolve_youtube_identity(urlinfo["direct_post_url"], **kwargs)
        elif urlinfo["url_class"].startswith("instagram_"):
            identity = resolve_instagram_identity(
                existing_creator_name=creator_name_existing,
                existing_creator_handle=existing_handle,
            )

    brand_link = verify_brand_link(
        brand=brand, video_title=video_title, caption_or_description=caption,
        channel_name=channel_name, creator_name=identity["creator_name"] or creator_name_existing,
        personalised_code=personalised_code, disclosure_type=disclosure_type,
    )

    partnership = classify_partnership(
        caption_or_description=caption, video_title=video_title,
        disclosure_type=disclosure_type, personalised_code=personalised_code,
    )

    has_content_text = bool(caption.strip() or video_title.strip())
    evidence_strength = compute_creator_evidence_strength(
        url_validation_status=urlinfo["url_validation_status"],
        creator_identity_confidence=identity["creator_identity_confidence"],
        brand_link_verified=brand_link["brand_link_verified"],
        has_content_text=has_content_text,
    )

    invalidation_reason = urlinfo["invalidation_reason"]
    usable = evidence_strength in ("strong", "medium")
    if urlinfo["url_validation_status"] != "valid":
        usable = False
    elif identity["creator_identity_confidence"] not in ("high", "medium"):
        usable = False
        invalidation_reason = "creator_identity_unresolved"
    elif not brand_link["brand_link_verified"]:
        usable = False
        invalidation_reason = "brand_link_unverified"
    else:
        invalidation_reason = ""

    out = {
        "url_class": urlinfo["url_class"],
        "direct_post_url": urlinfo["direct_post_url"],
        "direct_post_id": urlinfo["direct_post_id"],
        "url_validation_status": urlinfo["url_validation_status"],
        "invalidation_reason": invalidation_reason,
        "creator_handle": identity["creator_handle"],
        "creator_name": identity["creator_name"] or creator_name_existing,
        "creator_identity_source": identity["creator_identity_source"],
        "creator_identity_confidence": identity["creator_identity_confidence"],
        "creator_identity_evidence": identity["creator_identity_evidence"],
        "creator_identity_verified_at": NOW_ISO if identity["creator_identity_source"] != "none" else "",
        "brand_link_verified": brand_link["brand_link_verified"],
        "brand_link_evidence": brand_link["brand_link_evidence"],
        "brand_link_source": brand_link["brand_link_source"],
        "brand_link_confidence": brand_link["brand_link_confidence"],
        "partnership_type": partnership["partnership_type"],
        "partnership_evidence": partnership["partnership_evidence"],
        "partnership_confidence": partnership["partnership_confidence"],
        "partnership_inferred": partnership["partnership_inferred"],
        "evidence_strength": evidence_strength,
        "usable_as_creator_post": usable,
        "platform": platform,
    }
    return out


def dedupe(df: pd.DataFrame) -> tuple[pd.DataFrame, int, list]:
    """Deduplicate on (platform, direct_post_id); merge complementary metadata and
    keep the strongest defensible evidence_strength. Returns (deduped_df, n_merged,
    merge_log)."""
    strength_rank = {"strong": 3, "medium": 2, "weak": 1, "unusable": 0}
    groups: Dict[str, List[int]] = {}
    for idx, row in df.iterrows():
        key = f"{row['platform']}::{row['direct_post_id']}" if row.get("direct_post_id") else f"row::{idx}"
        groups.setdefault(key, []).append(idx)

    keep_rows = []
    merge_log = []
    n_merged = 0
    for key, idxs in groups.items():
        if len(idxs) == 1 or not key.split("::")[1]:
            keep_rows.append(df.loc[idxs[0]].to_dict())
            continue
        n_merged += len(idxs) - 1
        sub = df.loc[idxs]
        best_idx = sub["evidence_strength"].map(lambda s: strength_rank.get(s, -1)).idxmax()
        merged = df.loc[best_idx].to_dict()
        source_ids = sorted(set(str(df.loc[i].get("source_id", "")) for i in idxs if pd.notna(df.loc[i].get("source_id"))))
        merged["source_record_ids"] = "|".join(source_ids)
        # fill blanks in the kept row from sibling duplicates (non-conflicting merge)
        for i in idxs:
            for col in df.columns:
                cur = merged.get(col)
                cur_blank = cur is None or (isinstance(cur, float) and pd.isna(cur)) or str(cur).strip() == ""
                if cur_blank:
                    alt = df.loc[i].get(col)
                    if alt is not None and not (isinstance(alt, float) and pd.isna(alt)) and str(alt).strip():
                        merged[col] = alt
        keep_rows.append(merged)
        merge_log.append({
            "platform": df.loc[idxs[0]]["platform"], "direct_post_id": df.loc[idxs[0]]["direct_post_id"],
            "merged_row_count": len(idxs), "kept_evidence_strength": merged.get("evidence_strength"),
            "source_record_ids": merged.get("source_record_ids", ""),
        })

    out = pd.DataFrame(keep_rows).reset_index(drop=True)
    if "source_record_ids" not in out.columns:
        out["source_record_ids"] = ""
    out["source_record_ids"] = out["source_record_ids"].fillna("")
    return out, n_merged, merge_log


def replenish(df: pd.DataFrame, needed: int) -> pd.DataFrame:
    """Targeted replenishment via real DDG/YouTube discovery, run through the same
    classify_url/identity/brand-link/evidence-strength pipeline as existing rows.
    Never fabricates rows and never lowers verification thresholds."""
    from scripts.hydrate_day2_sources import _ddgs, CREATOR_DDG_QUERIES  # reuse existing discovery

    existing_urls = set(df["direct_post_url"].dropna().astype(str)) | set(
        df["source_url"].dropna().astype(str)
    )
    new_rows = []
    for brand in BRANDS:
        for query in CREATOR_DDG_QUERIES.get(brand, []):
            results = _ddgs(query, max_results=8)
            for r in results:
                href = r.get("href", "")
                if not href or href in existing_urls:
                    continue
                existing_urls.add(href)
                new_rows.append({
                    "brand": brand, "platform": (
                        "youtube" if "youtube" in href or "youtu.be" in href
                        else "tiktok" if "tiktok" in href else "instagram" if "instagram" in href else "other"
                    ),
                    "source_url": href, "post_url": href,
                    "video_title": r.get("title", ""), "caption_or_description": (r.get("title", "") + " " + r.get("body", ""))[:400],
                    "channel_name": "", "creator_name": "", "creator_handle": "",
                    "personalised_code": "", "disclosure_type": "",
                    "source_id": f"creator_repl_{len(new_rows)+1:04d}",
                    "source_method": "repair_replenishment", "collection_date": TODAY,
                    "collected_by": "auto_pipeline_day2_repair", "evidence_type": "creator_strategy",
                    "citation_ready": False, "synthetic": False, "notes": "",
                })
            if len(df) + len(new_rows) - needed >= 0:
                break
        candidate_count = sum(1 for _ in new_rows)
        if candidate_count >= needed * 2:  # gather a buffer; many will fail verification
            break

    if not new_rows:
        return df

    cand_df = pd.DataFrame(new_rows)
    for col in df.columns:
        if col not in cand_df.columns:
            cand_df[col] = None
    cand_df = cand_df[df.columns.tolist()]

    print(f"  [replenish] {len(cand_df)} new candidate URLs discovered via DDG search")
    repaired = []
    for _, row in cand_df.iterrows():
        fields = repair_row(row)
        merged = row.to_dict()
        merged.update(fields)
        repaired.append(merged)
    repaired_df = pd.DataFrame(repaired)
    n_verified_new = int((repaired_df["usable_as_creator_post"] == True).sum())
    print(f"  [replenish] {n_verified_new}/{len(repaired_df)} new candidates verified strong/medium")

    combined = pd.concat([df, repaired_df], ignore_index=True)
    combined, n_merged, _ = dedupe(combined)
    if n_merged:
        print(f"  [replenish] {n_merged} duplicate(s) consolidated against existing rows")
    return combined


def main() -> int:
    print(f"Loading {ENRICHED_PATH.relative_to(PROJECT_ROOT)} ...")
    df = pd.read_csv(ENRICHED_PATH)
    n_before = len(df)
    n_usable_before = int((df["usable_as_creator_post"] == True).sum())
    print(f"  {n_before} rows loaded, {n_usable_before} currently usable_as_creator_post=True")

    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    if ARCHIVE_PATH.exists():
        print(f"  Archive already exists, not overwriting: {ARCHIVE_PATH.relative_to(PROJECT_ROOT)}")
    else:
        df.to_csv(ARCHIVE_PATH, index=False)
        print(f"  Backed up pre-repair state to {ARCHIVE_PATH.relative_to(PROJECT_ROOT)}")

    print("\n[PART B-F] Classifying URLs, resolving identity, verifying brand link, "
          "classifying partnership, repairing evidence_strength ...")
    repaired_fields = []
    for i, (_, row) in enumerate(df.iterrows()):
        if (i + 1) % 15 == 0 or i == 0:
            print(f"  [{i+1}/{len(df)}]")
        repaired_fields.append(repair_row(row))
    repaired_df = df.copy()
    for col in REPAIR_COLUMNS:
        if col == "source_record_ids":
            continue
        repaired_df[col] = [f[col] for f in repaired_fields]
    repaired_df["platform"] = [f["platform"] for f in repaired_fields]
    repaired_df["source_record_ids"] = ""

    invalid_reasons = repaired_df.loc[
        repaired_df["usable_as_creator_post"] != True, "invalidation_reason"
    ].value_counts()
    print("\n  Invalidation reasons (rows now usable_as_creator_post=False):")
    for reason, n in invalid_reasons.items():
        label = reason if reason else "(none - passed all checks but strength not strong/medium)"
        print(f"    {label}: {n}")

    print("\n[PART G] Deduplicating on (platform, direct_post_id) ...")
    deduped_df, n_merged, merge_log = dedupe(repaired_df)
    print(f"  {len(repaired_df)} rows -> {len(deduped_df)} rows ({n_merged} duplicate(s) merged)")
    for m in merge_log:
        print(f"    merged {m['merged_row_count']} rows -> platform={m['platform']} "
              f"direct_post_id={m['direct_post_id']} kept_strength={m['kept_evidence_strength']}")

    n_verified = int((deduped_df["usable_as_creator_post"] == True).sum())
    print(f"\n  Verified strong/medium creator posts after repair: {n_verified}")

    if n_verified < VERIFIED_TARGET:
        print(f"\n[PART I] {n_verified} < target {VERIFIED_TARGET} -- running targeted replenishment ...")
        deduped_df = replenish(deduped_df, VERIFIED_TARGET - n_verified)
        n_verified = int((deduped_df["usable_as_creator_post"] == True).sum())
        print(f"  Verified strong/medium creator posts after replenishment: {n_verified}")
    else:
        print(f"  Target of {VERIFIED_TARGET} already met -- no replenishment needed.")

    deduped_df.to_csv(ENRICHED_PATH, index=False)
    print(f"\nWrote repaired file: {ENRICHED_PATH.relative_to(PROJECT_ROOT)} ({len(deduped_df)} rows)")

    print("\nBy brand x platform (verified strong/medium only):")
    verified = deduped_df[deduped_df["usable_as_creator_post"] == True]
    if not verified.empty:
        print(verified.groupby(["brand", "platform"]).size().to_string())
    else:
        print("  (none)")

    print(f"\nFinal: {n_before} -> {len(deduped_df)} rows, "
          f"{n_usable_before} -> {n_verified} verified usable_as_creator_post=True")
    print("\nNext: python scripts/recover_day2_gates.py")
    return 0 if n_verified >= VERIFIED_TARGET else 1


if __name__ == "__main__":
    sys.exit(main())
