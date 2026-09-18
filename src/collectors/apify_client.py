"""Apify REST API client for running public actors on the free tier."""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional

import requests


APIFY_BASE = "https://api.apify.com/v2"
DEFAULT_TIMEOUT_SECONDS = 300  # 5 minutes max per actor run
POLL_INTERVAL = 10  # seconds between status polls


class ApifyClient:
    """Thin Apify REST client (no SDK dependency)."""

    def __init__(self, token: str):
        self.token = token
        self.headers = {"Content-Type": "application/json"}

    @classmethod
    def from_env(cls) -> Optional["ApifyClient"]:
        """Build client from APIFY_TOKEN env var; return None if not set.

        Handles both raw token strings and full webhook URL format:
          apify_api_xxxxxx             -> used directly
          https://...?token=apify_api  -> token extracted
        """
        import os
        raw = os.getenv("APIFY_TOKEN", "").strip()
        if not raw:
            return None
        # Extract token from URL if stored as full URL
        if "token=" in raw:
            match = re.search(r"token=([A-Za-z0-9_]+)", raw)
            token = match.group(1) if match else ""
        elif raw.startswith("apify_api_"):
            token = raw
        else:
            token = raw
        return cls(token) if token else None

    def run_actor(
        self,
        actor_id: str,
        input_data: Dict[str, Any],
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> Optional[List[Dict[str, Any]]]:
        """Run an Apify actor, wait for completion, and return results list.

        Returns list of result items, or None on failure.
        """
        run_id, dataset_id = self._start_run(actor_id, input_data)
        if not run_id:
            return None

        status = self._wait_for_run(actor_id, run_id, timeout)
        if status != "SUCCEEDED":
            print(f"  [Apify] Actor {actor_id} finished with status: {status}")
            # Try to get partial results even on non-SUCCEEDED
            if not dataset_id:
                return None

        return self._get_dataset_items(dataset_id) if dataset_id else None

    def _start_run(self, actor_id: str, input_data: Dict) -> tuple[str, str]:
        """POST to start an actor run. Returns (run_id, dataset_id)."""
        url = f"{APIFY_BASE}/acts/{actor_id}/runs"
        params = {"token": self.token}
        try:
            resp = requests.post(url, json=input_data, params=params,
                                 headers=self.headers, timeout=30)
            resp.raise_for_status()
            data = resp.json().get("data", {})
            run_id = data.get("id", "")
            dataset_id = data.get("defaultDatasetId", "")
            print(f"  [Apify] Started actor {actor_id}, run={run_id}")
            return run_id, dataset_id
        except Exception as exc:
            print(f"  [Apify] Failed to start {actor_id}: {exc}")
            return "", ""

    def _wait_for_run(self, actor_id: str, run_id: str, timeout: int) -> str:
        """Poll run status until terminal state. Returns final status string."""
        url = f"{APIFY_BASE}/acts/{actor_id}/runs/{run_id}"
        params = {"token": self.token}
        terminal = {"SUCCEEDED", "FAILED", "ABORTED", "TIMED-OUT"}
        elapsed = 0
        while elapsed < timeout:
            time.sleep(POLL_INTERVAL)
            elapsed += POLL_INTERVAL
            try:
                resp = requests.get(url, params=params, timeout=15)
                resp.raise_for_status()
                status = resp.json().get("data", {}).get("status", "UNKNOWN")
                print(f"  [Apify] Run {run_id}: {status} ({elapsed}s)")
                if status in terminal:
                    return status
            except Exception:
                pass
        return "TIMED-OUT"

    def _get_dataset_items(self, dataset_id: str) -> List[Dict[str, Any]]:
        """Download all items from an Apify dataset."""
        url = f"{APIFY_BASE}/datasets/{dataset_id}/items"
        params = {"token": self.token, "format": "json", "limit": 1000}
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json() or []
        except Exception as exc:
            print(f"  [Apify] Failed to fetch dataset {dataset_id}: {exc}")
            return []


# ── Convenience runners ───────────────────────────────────────────────────────

def run_trustpilot_scraper(
    client: ApifyClient,
    domain: str,
    count: int = 50,
) -> List[Dict[str, Any]]:
    """Run theagents~trustpilot-reviews for a given domain.

    Uses startUrls + maxReviews format. Returns raw Apify result items.
    domain: bare domain like 'wearetala.com' or 'adanola.com'.
    """
    actor_id = "theagents~trustpilot-reviews"
    url = f"https://www.trustpilot.com/review/{domain}"
    input_data = {
        "startUrls": [{"url": url}],
        "maxReviews": count,
        "reviewsLanguage": "en",
    }
    print(f"  [Apify] Scraping Trustpilot: {url} ({count} reviews)")
    return client.run_actor(actor_id, input_data) or []


def run_google_search_scraper(
    client: ApifyClient,
    queries: List[str],
    results_per_query: int = 10,
) -> List[Dict[str, Any]]:
    """Run apify/google-search-scraper for a list of queries.

    Returns raw Apify result items.
    """
    actor_id = "apify/google-search-scraper"
    input_data = {
        "queries": "\n".join(queries),
        "maxPagesPerQuery": 1,
        "resultsPerPage": results_per_query,
        "countryCode": "gb",
        "languageCode": "en",
    }
    print(f"  [Apify] Google search: {len(queries)} queries")
    return client.run_actor(actor_id, input_data) or []
