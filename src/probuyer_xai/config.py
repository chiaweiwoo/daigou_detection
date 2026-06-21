"""Central config: paths, feature lists, thresholds, env loading."""

from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

# --- Repository root ---------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]

# --- Directories -------------------------------------------------------------
DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
RULES_DIR = ROOT / "rules"
REPORTS_DIR = ROOT / "reports"

# --- Data files --------------------------------------------------------------
RAW_XLSX = DATA_RAW / "Online Retail.xlsx"
TRANSACTIONS_CLEAN = DATA_PROCESSED / "transactions_clean.parquet"
TRANSACTIONS_NORMAL = DATA_PROCESSED / "transactions_normal.parquet"
CUSTOMER_FEATURES = DATA_PROCESSED / "customer_features.parquet"
CUSTOMER_SCORES = DATA_PROCESSED / "customer_scores.parquet"
CUSTOMER_RULE_HITS = DATA_PROCESSED / "customer_rule_hits.parquet"
FEATURE_METADATA = DATA_PROCESSED / "feature_metadata.json"

# --- Model files -------------------------------------------------------------
HBOS_MODEL = MODELS_DIR / "hbos_model.joblib"
FEATURE_SCALER = MODELS_DIR / "feature_scaler.joblib"
MODEL_METADATA = MODELS_DIR / "model_metadata.json"

# --- Rules -------------------------------------------------------------------
PROBUYER_RULES = RULES_DIR / "probuyer_rules.json"

# --- Reports -----------------------------------------------------------------
REPORT_DATA = REPORTS_DIR / "data_understanding.md"
REPORT_MODEL = REPORTS_DIR / "model_summary.md"
REPORT_CASES = REPORTS_DIR / "case_studies.md"
REPORT_LLM = REPORTS_DIR / "llm_examples.md"

# --- Feature lists -----------------------------------------------------------
MODEL_FEATURES = [
    "total_quantity",
    "total_spend",
    "num_invoices",
    "active_days",
    "avg_quantity_per_invoice",
    "max_quantity_per_invoice",
    "avg_spend_per_invoice",
    "max_spend_per_invoice",
    "unique_skus",
    "repeat_sku_ratio",
    "top_sku_quantity_share",
    "top_sku_spend_share",
    "quantity_per_active_day",
    "spend_per_active_day",
    "bulk_invoice_count",
    "bulk_invoice_ratio",
    "cancellation_ratio",
]

# --- Risk band thresholds (percentile of anomaly score) ----------------------
RISK_HIGH_PCT = 99.0   # top 1%  → High
RISK_MED_PCT = 97.0    # top 1–3% → Medium  (everything else → Low)

# --- LLM env -----------------------------------------------------------------
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
