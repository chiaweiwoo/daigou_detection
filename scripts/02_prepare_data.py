"""Clean raw data and save processed parquets + data understanding report."""

from probuyer_xai.data import load_raw, clean, save_processed
from probuyer_xai.config import REPORT_DATA, REPORTS_DIR

import textwrap


def _report(txn_clean, txn_normal, raw_df):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    cancelled = txn_clean[txn_clean["is_cancelled"]]
    top_countries = (
        txn_normal.groupby("country")["amount"].sum().sort_values(ascending=False).head(10)
    )
    qty_desc = txn_normal["quantity"].describe()
    amt_desc = txn_normal["amount"].describe()

    lines = [
        "# Data Understanding Report",
        "",
        "## Overview",
        f"- Raw rows (with customer_id): {len(raw_df):,}",
        f"- Clean rows (all with customer_id): {len(txn_clean):,}",
        f"- Normal purchase rows (qty>0, price>0): {len(txn_normal):,}",
        f"- Unique customers: {txn_normal['customer_id'].nunique():,}",
        f"- Unique invoices: {txn_normal['invoice_no'].nunique():,}",
        f"- Date range: {txn_normal['invoice_date'].min().date()} to {txn_normal['invoice_date'].max().date()}",
        "",
        "## Data quality",
        f"- Rows dropped (missing customer_id): {len(raw_df) - len(txn_clean):,}",
        f"- Cancelled / return rows: {len(cancelled):,}",
        "",
        "## Top 10 countries by spend",
        "| Country | Total Spend |",
        "|---|---|",
    ]
    for country, spend in top_countries.items():
        lines.append(f"| {country} | {spend:,.2f} |")

    lines += [
        "",
        "## Quantity distribution (normal purchases)",
        f"- Mean: {qty_desc['mean']:.1f}",
        f"- Median: {qty_desc['50%']:.1f}",
        f"- p95: {txn_normal['quantity'].quantile(0.95):.1f}",
        f"- Max: {qty_desc['max']:.0f}",
        "",
        "## Amount distribution (normal purchases)",
        f"- Mean: {amt_desc['mean']:.2f}",
        f"- Median: {amt_desc['50%']:.2f}",
        f"- p95: {txn_normal['amount'].quantile(0.95):.2f}",
        f"- Max: {amt_desc['max']:.2f}",
    ]

    REPORT_DATA.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report saved: {REPORT_DATA}")


if __name__ == "__main__":
    raw_df = load_raw()
    txn_clean, txn_normal = clean(raw_df)
    save_processed(txn_clean, txn_normal)
    _report(txn_clean, txn_normal, raw_df)
    print("Phase 2 complete.")
