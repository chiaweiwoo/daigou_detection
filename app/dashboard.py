"""Streamlit dashboard — Retail Probuyer Risk Detection."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --- Setup page ---
st.set_page_config(
    page_title="Probuyer Risk Dashboard",
    page_icon="🔍",
    layout="wide",
)

# --- Helpers -----------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"
RULES = ROOT / "rules"
REPORTS = ROOT / "reports"


@st.cache_data
def _load_features():
    p = DATA / "customer_features.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p)


@st.cache_data
def _load_scores():
    p = DATA / "customer_scores.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p)


@st.cache_data
def _load_rule_hits():
    p = DATA / "customer_rule_hits.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p)


@st.cache_data
def _load_normal():
    p = DATA / "transactions_normal.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    return df


@st.cache_data
def _load_model_meta():
    p = MODELS / "model_metadata.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


@st.cache_data
def _load_rules_doc():
    p = RULES / "probuyer_rules.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def _build_evidences(features, scores, rule_hits):
    from probuyer_xai.explain import build_evidence
    return build_evidence(features, scores, rule_hits)


def _check_artifacts():
    missing = []
    for p in [
        DATA / "customer_features.parquet",
        DATA / "customer_scores.parquet",
        DATA / "customer_rule_hits.parquet",
    ]:
        if not p.exists():
            missing.append(str(p))
    return missing


# --- Main app ----------------------------------------------------------------

def main():
    st.sidebar.title("Probuyer Risk Detection")
    st.sidebar.caption("Retail anomaly detection with LLM explainability")

    missing = _check_artifacts()
    if missing:
        st.error(
            "Pipeline artifacts not found. Run the pipeline first:\n\n"
            "```\nuv run python scripts/06_run_all.py\n```"
        )
        st.stop()

    page = st.sidebar.radio(
        "Navigate",
        ["Overview", "Customer Investigation", "What-if Simulator", "Case Studies", "Model Calibration"],
    )

    features = _load_features()
    scores = _load_scores()
    rule_hits = _load_rule_hits()

    if page == "Overview":
        _page_overview(features, scores, rule_hits)
    elif page == "Customer Investigation":
        _page_investigation(features, scores, rule_hits)
    elif page == "What-if Simulator":
        _page_whatif(features, scores, rule_hits)
    elif page == "Case Studies":
        _page_case_studies()
    elif page == "Model Calibration":
        _page_calibration()


# --- Page 1: Overview -------------------------------------------------------

def _page_overview(features, scores, rule_hits):
    st.title("Overview")

    total = len(scores)
    high = int((scores["risk_band"] == "High").sum())
    med = int((scores["risk_band"] == "Medium").sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("Total customers", f"{total:,}")
    c2.metric("High risk", f"{high:,}", f"{high/total:.1%}")
    c3.metric("Medium risk", f"{med:,}", f"{med/total:.1%}")

    st.subheader("Anomaly score distribution")
    fig = px.histogram(
        scores,
        x="anomaly_score",
        color="risk_band",
        color_discrete_map={"High": "#e74c3c", "Medium": "#f39c12", "Low": "#2ecc71"},
        nbins=60,
        labels={"anomaly_score": "Anomaly Score", "risk_band": "Risk Band"},
    )
    st.plotly_chart(fig, use_container_width=True)

    evidences = _build_evidences(features, scores, rule_hits)

    st.subheader("Anomaly type breakdown")
    from collections import Counter
    type_counts = Counter(e["anomaly_type"] for e in evidences)
    fig2 = px.pie(
        values=list(type_counts.values()),
        names=[k.replace("_", " ").title() for k in type_counts.keys()],
        hole=0.4,
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Top 20 flagged customers")
    merged = scores.merge(
        rule_hits[["customer_id", "rule_count"]],
        on="customer_id",
        how="left",
    )
    merged["rule_count"] = merged["rule_count"].fillna(0).astype(int)
    top20 = merged.nlargest(20, "anomaly_score")[
        ["customer_id", "risk_band", "risk_percentile", "anomaly_score", "rule_count"]
    ]
    st.dataframe(
        top20.rename(columns={
            "customer_id": "Customer ID",
            "risk_band": "Risk Band",
            "risk_percentile": "Percentile",
            "anomaly_score": "Anomaly Score",
            "rule_count": "Rule Hits",
        }),
        use_container_width=True,
        hide_index=True,
    )


# --- Page 2: Customer Investigation -----------------------------------------

def _page_investigation(features, scores, rule_hits):
    st.title("Customer Investigation")

    high_med = scores[scores["risk_band"].isin(["High", "Medium"])].sort_values(
        "anomaly_score", ascending=False
    )
    cid = st.selectbox(
        "Select customer",
        options=high_med["customer_id"].tolist(),
        format_func=lambda x: f"{x} ({scores.loc[scores['customer_id'] == x, 'risk_band'].iloc[0]})",
    )

    evidences = _build_evidences(features, scores, rule_hits)
    from probuyer_xai.explain import evidence_for_customer
    ev = evidence_for_customer(str(cid), evidences)

    if ev is None:
        st.warning("No evidence found for this customer.")
        return

    c1, c2, c3, c4 = st.columns(4)
    band_color = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
    c1.metric("Risk Band", f"{band_color.get(ev['risk_band'], '')} {ev['risk_band']}")
    c2.metric("Risk Percentile", f"{ev['risk_percentile']:.1f}")
    c3.metric("Confidence", ev["confidence"])
    c4.metric("Rule Hits", ev["rule_count"])

    st.subheader("Anomaly type")
    st.info(ev["anomaly_type"].replace("_", " ").title())

    st.subheader("Key metrics")
    m = ev.get("key_metrics", {})
    metric_df = pd.DataFrame(
        {
            "Metric": [
                "Total quantity",
                "Total spend",
                "Invoices",
                "Max invoice qty",
                "Bulk invoice ratio",
                "Top SKU qty share",
                "Cancellation ratio",
            ],
            "Value": [
                f"{m.get('total_quantity', 0):,.0f}",
                f"{m.get('total_spend', 0):,.2f}",
                f"{m.get('num_invoices', 0)}",
                f"{m.get('max_quantity_per_invoice', 0):,.0f}",
                f"{m.get('bulk_invoice_ratio', 0):.2%}",
                f"{m.get('top_sku_quantity_share', 0):.2%}",
                f"{m.get('cancellation_ratio', 0):.2%}",
            ],
        }
    )
    st.dataframe(metric_df, use_container_width=True, hide_index=True)

    st.subheader("Rule hits")
    if ev["rule_hits"]:
        rules_doc = _load_rules_doc()
        rule_map = {r["rule_id"]: r for r in rules_doc.get("rules", [])}
        for rid in ev["rule_hits"]:
            r = rule_map.get(rid, {})
            st.warning(f"**{rid}**: {r.get('name', rid)} — {r.get('risk_reason', '')}")
    else:
        st.success("No rule hits (anomaly detected by model only).")

    st.subheader("Business reasons")
    for reason in ev.get("top_reasons", []):
        st.write(f"- {reason}")

    st.subheader("LLM explanation")
    with st.spinner("Generating explanation ..."):
        from probuyer_xai.llm import explain_customer
        explanation = explain_customer(ev)
    st.info(explanation)

    st.subheader("Recommended action")
    st.write(ev.get("recommended_action", ""))

    st.subheader("Sample transactions")
    txn = _load_normal()
    if txn is not None:
        cust_txn = txn[txn["customer_id"] == str(cid)].sort_values(
            "invoice_date", ascending=False
        ).head(20)
        st.dataframe(
            cust_txn[["invoice_date", "invoice_no", "stock_code", "description", "quantity", "unit_price", "amount"]],
            use_container_width=True,
            hide_index=True,
        )


# --- Page 3: What-if Simulator ----------------------------------------------

def _page_whatif(features, scores, rule_hits):
    st.title("What-if Simulator")
    st.caption("Adjust detection parameters and see the impact — Python computes the result, LLM explains it.")

    evidences = _build_evidences(features, scores, rule_hits)

    col1, col2 = st.columns(2)
    with col1:
        scenario = st.radio(
            "Scenario",
            [
                "Change risk threshold",
                "Exclude cancellation anomalies",
                "Whitelist a customer",
                "Require minimum rule hits",
            ],
        )
        if scenario == "Change risk threshold":
            new_high = st.slider("High-risk percentile threshold", 95.0, 99.9, 99.0, 0.1)
            new_med = st.slider("Medium-risk percentile threshold", 90.0, new_high - 0.1, 97.0, 0.1)
        elif scenario == "Whitelist a customer":
            wl_cid = st.text_input("Customer ID to whitelist")
        elif scenario == "Require minimum rule hits":
            min_hits = st.slider("Minimum rule hits", 1, 5, 2)

        exclude_cancel = False
        if scenario == "Exclude cancellation anomalies":
            exclude_cancel = True

    with col2:
        if st.button("Run scenario", type="primary"):
            from probuyer_xai.whatif import (
                change_risk_threshold,
                exclude_cancellation_anomalies,
                whitelist_customer,
                require_min_rule_hits,
            )
            from probuyer_xai.llm import explain_whatif

            if scenario == "Change risk threshold":
                result = change_risk_threshold(scores, evidences, new_high, new_med)
            elif scenario == "Exclude cancellation anomalies":
                result = exclude_cancellation_anomalies(scores, evidences)
            elif scenario == "Whitelist a customer":
                result = whitelist_customer(wl_cid or "0", scores, evidences)
            else:
                result = require_min_rule_hits(min_hits, scores, evidences)

            st.metric("Before flagged", result["before_flagged_count"])
            st.metric("After flagged", result["after_flagged_count"])
            delta = result["after_flagged_count"] - result["before_flagged_count"]
            st.metric("Change", f"{delta:+d}")

            if result["customers_added"]:
                st.success(f"Customers added ({len(result['customers_added'])}): {', '.join(result['customers_added'][:10])}")
            if result["customers_removed"]:
                st.info(f"Customers removed ({len(result['customers_removed'])}): {', '.join(result['customers_removed'][:10])}")

            st.subheader("LLM interpretation")
            with st.spinner("Generating ..."):
                interp = explain_whatif(result)
            st.write(interp)


# --- Page 4: Case Studies ---------------------------------------------------

def _page_case_studies():
    st.title("Case Studies")
    p = REPORTS / "case_studies.md"
    if not p.exists():
        st.error(
            "Case studies not yet generated. Run:\n\n"
            "```\nuv run python scripts/05_generate_case_studies.py\n```"
        )
        return
    st.markdown(p.read_text(encoding="utf-8"))


# --- Page 5: Model Calibration ----------------------------------------------

def _page_calibration():
    st.title("Model Calibration")
    st.caption(
        "An independent pro LLM rates sampled customers on probuyer-likeness (1–5, temperature=0). "
        "Agreement between the LLM's score and the model's risk band is the calibration signal."
    )

    from probuyer_xai.calibrate import load_report

    report = load_report()

    if report is None:
        st.info(
            "No calibration report found. Run:\n\n"
            "```\nuv run python scripts/07_calibrate_model.py\n```"
        )
        return

    mode_label = "Mock (no API key)" if report.is_mock else report.model_used
    st.caption(f"Iteration {report.iteration} · {report.timestamp[:10]} · Model: {mode_label}")

    # Overall agreement metric
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Overall agreement", f"{report.overall_agreement:.1%}")
    for col, bc in zip([col2, col3, col4], report.by_band):
        col.metric(f"{bc.band} band", f"{bc.agreement_rate:.1%}", f"{bc.agreed}/{bc.sampled} rated")

    # Per-band agreement bar chart
    st.subheader("Agreement by risk band")
    band_data = {
        "Band": [bc.band for bc in report.by_band],
        "Agreement": [bc.agreement_rate for bc in report.by_band],
        "Sampled": [bc.sampled for bc in report.by_band],
    }
    fig = px.bar(
        band_data,
        x="Band",
        y="Agreement",
        color="Band",
        color_discrete_map={"High": "#e74c3c", "Medium": "#f39c12", "Low": "#2ecc71"},
        text=[f"{r:.0%}" for r in band_data["Agreement"]],
        range_y=[0, 1],
        labels={"Agreement": "Agreement Rate"},
    )
    fig.update_traces(textposition="outside")
    fig.add_hline(y=0.7, line_dash="dash", line_color="gray", annotation_text="70% target")
    st.plotly_chart(fig, use_container_width=True)

    # LLM score distribution
    st.subheader("LLM score distribution")
    score_data = pd.DataFrame([
        {"Band": r.model_band, "LLM Score": r.llm_score, "Agrees": r.agrees_with_model}
        for r in report.raw_ratings
    ])
    if not score_data.empty:
        fig2 = px.histogram(
            score_data,
            x="LLM Score",
            color="Band",
            color_discrete_map={"High": "#e74c3c", "Medium": "#f39c12", "Low": "#2ecc71"},
            barmode="group",
            nbins=5,
            range_x=[0.5, 5.5],
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Suggested adjustments
    st.subheader("Suggested adjustments")
    for s in report.suggested_adjustments:
        st.warning(s)

    # Disagreement cases
    all_disagreements = [r for bc in report.by_band for r in bc.disagreements]
    if all_disagreements:
        st.subheader(f"Disagreement cases ({len(all_disagreements)})")
        st.caption("Customers where the LLM's probuyer rating diverges from the model's risk band.")
        rows = []
        for r in all_disagreements:
            rows.append({
                "Customer ID": r.customer_id,
                "Model band": r.model_band,
                "LLM score": r.llm_score,
                "Primary signal": r.primary_signal or "—",
                "Disqualifiers": "; ".join(r.disqualifiers[:2]),
                "LLM confidence": r.rating_confidence,
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.success("No disagreement cases — all sampled customers agree with the model's rating.")


if __name__ == "__main__":
    main()
