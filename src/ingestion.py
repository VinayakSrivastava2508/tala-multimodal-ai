"""Data ingestion utilities for TALA Multimodal AI project."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
import yaml
from bs4 import BeautifulSoup


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
CONFIGS = PROJECT_ROOT / "configs"


def load_brands_config() -> dict:
    """Return parsed brands.yaml as a dict."""
    with open(CONFIGS / "brands.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_schema_config() -> dict:
    """Return parsed schema.yaml as a dict."""
    with open(CONFIGS / "schema.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_csv(filename: str, subfolder: str = "raw") -> pd.DataFrame:
    """Load a CSV from data/{subfolder}/{filename} and return a DataFrame."""
    path = PROJECT_ROOT / "data" / subfolder / filename
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Create the file using the manual CSV template."
        )
    return pd.read_csv(path, parse_dates=True, low_memory=False)


def save_csv(df: pd.DataFrame, filename: str, subfolder: str = "interim") -> Path:
    """Save DataFrame to data/{subfolder}/{filename} and return the path."""
    out_dir = PROJECT_ROOT / "data" / subfolder
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    df.to_csv(out_path, index=False)
    return out_path


def fetch_page(url: str, delay: float = 2.0, timeout: int = 10) -> Optional[BeautifulSoup]:
    """Fetch a public URL with a polite delay and return a BeautifulSoup object.

    Returns None if the request fails. Respects robots.txt convention by
    sleeping `delay` seconds before each request.
    """
    time.sleep(delay)
    headers = {"User-Agent": "tala-research-bot/0.1 (academic; non-commercial)"}
    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        return BeautifulSoup(response.text, "lxml")
    except requests.RequestException as exc:
        print(f"[ingestion] Request failed for {url}: {exc}")
        return None


def create_empty_template(schema_name: str, brand_slug: str = "") -> pd.DataFrame:
    """Create an empty DataFrame with columns from schema.yaml for manual data entry.

    Returns an empty DataFrame with the correct column names.
    Saves it to data/raw/{schema_name}_{brand_slug}.csv if brand_slug provided.
    """
    schema = load_schema_config()
    if schema_name not in schema["schemas"]:
        raise ValueError(f"Unknown schema: {schema_name}. Check configs/schema.yaml.")
    columns = list(schema["schemas"][schema_name]["fields"].keys())
    df = pd.DataFrame(columns=columns)
    if brand_slug:
        fname = f"{schema_name}_{brand_slug}.csv" if brand_slug else f"{schema_name}.csv"
        save_csv(df, fname, subfolder="raw")
        print(f"[ingestion] Template saved → data/raw/{fname}")
    return df


def list_available_datasets() -> list[str]:
    """Return filenames of all CSVs currently in data/raw/."""
    return [p.name for p in DATA_RAW.glob("*.csv")]
