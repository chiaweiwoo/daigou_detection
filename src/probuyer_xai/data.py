"""Data loading and cleaning for UCI Online Retail."""

from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd

from probuyer_xai.config import (
    RAW_XLSX,
    TRANSACTIONS_CLEAN,
    TRANSACTIONS_NORMAL,
    DATA_PROCESSED,
)

_UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "00352/Online%20Retail.xlsx"
)

_COL_MAP = {
    "InvoiceNo": "invoice_no",
    "StockCode": "stock_code",
    "Description": "description",
    "Quantity": "quantity",
    "InvoiceDate": "invoice_date",
    "UnitPrice": "unit_price",
    "CustomerID": "customer_id",
    "Country": "country",
}


def download_raw(dest: Path = RAW_XLSX) -> bool:
    """Try to download the UCI Online Retail xlsx.

    Returns True if the file is ready (already existed or just downloaded).
    Returns False if the download fails.
    """
    if dest.exists():
        print(f"Raw file already exists: {dest}")
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading UCI Online Retail to {dest} ...")
    try:
        urllib.request.urlretrieve(_UCI_URL, dest)
        print("Download complete.")
        return True
    except Exception as exc:
        print(f"Download failed: {exc}")
        if dest.exists():
            dest.unlink()
        return False


def load_raw(path: Path = RAW_XLSX) -> pd.DataFrame:
    """Load the raw xlsx and standardise column names."""
    df = pd.read_excel(path, dtype={"CustomerID": str, "InvoiceNo": str, "StockCode": str})
    df = df.rename(columns=_COL_MAP)
    df["customer_id"] = df["customer_id"].astype(str).str.strip()
    df["invoice_no"] = df["invoice_no"].astype(str).str.strip()
    df["stock_code"] = df["stock_code"].astype(str).str.strip()
    return df


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean the raw DataFrame.

    Returns:
        transactions_clean: all rows with customer_id, parsed date, amount,
            and a ``is_cancelled`` flag.
        transactions_normal: subset with qty>0, price>0 (valid purchases).
    """
    # Drop rows without customer_id
    df = df[df["customer_id"].notna() & (df["customer_id"] != "nan")].copy()

    # Parse dates
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])

    # Derived amount
    df["amount"] = df["quantity"] * df["unit_price"]

    # Cancellation flag: invoice starts with C or negative quantity
    df["is_cancelled"] = df["invoice_no"].str.startswith("C") | (df["quantity"] < 0)

    transactions_clean = df.copy()

    transactions_normal = df[
        (~df["is_cancelled"]) & (df["quantity"] > 0) & (df["unit_price"] > 0)
    ].copy()

    return transactions_clean, transactions_normal


def save_processed(
    transactions_clean: pd.DataFrame,
    transactions_normal: pd.DataFrame,
) -> None:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    transactions_clean.to_parquet(TRANSACTIONS_CLEAN, index=False)
    transactions_normal.to_parquet(TRANSACTIONS_NORMAL, index=False)
    print(f"Saved {len(transactions_clean):,} rows -> {TRANSACTIONS_CLEAN}")
    print(f"Saved {len(transactions_normal):,} rows -> {TRANSACTIONS_NORMAL}")


def load_clean() -> pd.DataFrame:
    return pd.read_parquet(TRANSACTIONS_CLEAN)


def load_normal() -> pd.DataFrame:
    return pd.read_parquet(TRANSACTIONS_NORMAL)
