"""Customer-level feature engineering from cleaned transactions."""

from __future__ import annotations

import json
import numpy as np
import pandas as pd

from probuyer_xai.config import (
    CUSTOMER_FEATURES,
    FEATURE_METADATA,
    DATA_PROCESSED,
    TRANSACTIONS_CLEAN,
    TRANSACTIONS_NORMAL,
)

# Percentile used to define a "bulk" invoice
_BULK_QTY_PCT = 95


def _invoice_qty(txn_normal: pd.DataFrame) -> pd.Series:
    """Total quantity per invoice across all normal purchases."""
    return txn_normal.groupby("invoice_no")["quantity"].sum()


def build_features(
    txn_clean: pd.DataFrame,
    txn_normal: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Build one row per customer.

    Returns:
        features: DataFrame with customer_id index.
        metadata: dict with bulk threshold and feature list.
    """
    # --- Invoice-level bulk threshold ---
    inv_qty = _invoice_qty(txn_normal)
    bulk_threshold = float(np.percentile(inv_qty.values, _BULK_QTY_PCT))

    # Mark bulk invoices
    bulk_invoices = set(inv_qty[inv_qty >= bulk_threshold].index)
    txn_n = txn_normal.copy()
    txn_n["is_bulk"] = txn_n["invoice_no"].isin(bulk_invoices)

    # --- Normal-purchase aggregations ---
    grp = txn_n.groupby("customer_id")

    total_quantity = grp["quantity"].sum().rename("total_quantity")
    total_spend = grp["amount"].sum().rename("total_spend")

    inv_grp = txn_n.groupby(["customer_id", "invoice_no"])
    inv_qty_by_cust = inv_grp["quantity"].sum().reset_index()
    inv_spend_by_cust = inv_grp["amount"].sum().reset_index()

    num_invoices = inv_qty_by_cust.groupby("customer_id")["invoice_no"].count().rename("num_invoices")

    active_days = (
        grp["invoice_date"]
        .agg(lambda x: max((x.max() - x.min()).days + 1, 1))
        .rename("active_days")
    )

    avg_qty_per_inv = (
        inv_qty_by_cust.groupby("customer_id")["quantity"].mean().rename("avg_quantity_per_invoice")
    )
    max_qty_per_inv = (
        inv_qty_by_cust.groupby("customer_id")["quantity"].max().rename("max_quantity_per_invoice")
    )
    avg_spend_per_inv = (
        inv_spend_by_cust.groupby("customer_id")["amount"].mean().rename("avg_spend_per_invoice")
    )
    max_spend_per_inv = (
        inv_spend_by_cust.groupby("customer_id")["amount"].max().rename("max_spend_per_invoice")
    )

    unique_skus = grp["stock_code"].nunique().rename("unique_skus")

    # Descriptions (some SKUs share descriptions; use description count as proxy)
    unique_descriptions = grp["description"].nunique().rename("unique_descriptions")

    # repeat_sku_ratio: fraction of SKUs purchased more than once
    sku_counts = txn_n.groupby(["customer_id", "stock_code"])["quantity"].count().reset_index()
    repeat_sku = (
        sku_counts[sku_counts["quantity"] > 1]
        .groupby("customer_id")["stock_code"]
        .count()
        .rename("repeat_skus")
    )
    repeat_sku_ratio = (repeat_sku / unique_skus).rename("repeat_sku_ratio").fillna(0).clip(0, 1)

    # top_sku_quantity_share
    sku_qty = txn_n.groupby(["customer_id", "stock_code"])["quantity"].sum().reset_index()
    top_sku_qty = sku_qty.groupby("customer_id")["quantity"].max().rename("top_sku_qty")
    top_sku_quantity_share = (top_sku_qty / total_quantity).rename("top_sku_quantity_share").clip(0, 1)

    # top_sku_spend_share
    sku_spend = txn_n.groupby(["customer_id", "stock_code"])["amount"].sum().reset_index()
    top_sku_spend = sku_spend.groupby("customer_id")["amount"].max().rename("top_sku_spend")
    top_sku_spend_share = (top_sku_spend / total_spend).rename("top_sku_spend_share").clip(0, 1)

    quantity_per_active_day = (total_quantity / active_days).rename("quantity_per_active_day")
    spend_per_active_day = (total_spend / active_days).rename("spend_per_active_day")

    # Bulk invoice features
    bulk_inv_cust = txn_n[txn_n["is_bulk"]].groupby("customer_id")["invoice_no"].nunique().rename("bulk_invoice_count")
    bulk_invoice_count = bulk_inv_cust.reindex(num_invoices.index, fill_value=0)
    bulk_invoice_ratio = (bulk_invoice_count / num_invoices).rename("bulk_invoice_ratio").clip(0, 1)

    # --- Cancellation features (from transactions_clean) ---
    cancelled = txn_clean[txn_clean["is_cancelled"]]
    cancel_grp = cancelled.groupby("customer_id")

    cancelled_invoice_count = cancel_grp["invoice_no"].nunique().rename("cancelled_invoice_count")
    cancelled_quantity_abs = cancel_grp["quantity"].apply(lambda x: x.abs().sum()).rename("cancelled_quantity_abs")
    cancelled_amount_abs = cancel_grp["amount"].apply(lambda x: x.abs().sum()).rename("cancelled_amount_abs")

    cancellation_ratio = (
        cancelled_invoice_count / num_invoices
    ).rename("cancellation_ratio").clip(0, 1)

    # --- Date features ---
    first_purchase = grp["invoice_date"].min().rename("first_purchase_date")
    last_purchase = grp["invoice_date"].max().rename("last_purchase_date")

    # --- Assemble ---
    features = pd.concat(
        [
            total_quantity,
            total_spend,
            num_invoices,
            active_days,
            avg_qty_per_inv,
            max_qty_per_inv,
            avg_spend_per_inv,
            max_spend_per_inv,
            unique_skus,
            unique_descriptions,
            repeat_sku_ratio,
            top_sku_quantity_share,
            top_sku_spend_share,
            quantity_per_active_day,
            spend_per_active_day,
            bulk_invoice_count,
            bulk_invoice_ratio,
            cancelled_invoice_count,
            cancelled_quantity_abs,
            cancelled_amount_abs,
            cancellation_ratio,
            first_purchase,
            last_purchase,
        ],
        axis=1,
    )

    # Fill cancellation NaNs with 0 for customers with no cancellations
    cancel_cols = [
        "cancelled_invoice_count",
        "cancelled_quantity_abs",
        "cancelled_amount_abs",
        "cancellation_ratio",
    ]
    features[cancel_cols] = features[cancel_cols].fillna(0)

    # Guard against inf
    numeric_cols = features.select_dtypes(include="number").columns
    features[numeric_cols] = features[numeric_cols].replace([np.inf, -np.inf], np.nan)

    # Drop customers with any NaN in model features (shouldn't happen but safety net)
    from probuyer_xai.config import MODEL_FEATURES
    before = len(features)
    features = features.dropna(subset=MODEL_FEATURES)
    if len(features) < before:
        print(f"Dropped {before - len(features)} customers with NaN model features.")

    metadata = {
        "bulk_threshold_qty": bulk_threshold,
        "bulk_pct": _BULK_QTY_PCT,
        "n_customers": len(features),
        "feature_columns": list(features.columns),
    }

    return features, metadata


def save_features(features: pd.DataFrame, metadata: dict) -> None:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    features.to_parquet(CUSTOMER_FEATURES)
    FEATURE_METADATA.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    print(f"Saved {len(features):,} customer features -> {CUSTOMER_FEATURES}")
    print(f"Feature metadata -> {FEATURE_METADATA}")


def load_features() -> pd.DataFrame:
    return pd.read_parquet(CUSTOMER_FEATURES)
