from pathlib import Path
import json

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).parent
FEATURE_NAMES = {
    "cycle": "Time already in service",
    "setting_1": "Operating condition A",
    "setting_2": "Operating condition B",
    "setting_3": "Operating condition C",
    **{f"sensor_{i}": f"Engine condition signal {i}" for i in range(1, 22)},
}

st.set_page_config(page_title="Aircraft Engine Maintenance Forecast", page_icon="✈️", layout="wide")
st.title("Aircraft Engine Maintenance Forecast")
st.caption("A maintenance-planning demo that estimates how much operating time an aircraft engine has left.")
st.info("**What is a cycle?** In this dataset, one cycle is one recorded period of engine operation. More remaining cycles means more estimated time before maintenance is needed.")

metrics_path = ROOT / "artifacts" / "metrics.json"
if not metrics_path.exists():
    st.error("The trained model is missing. Run `python train.py` first.")
    st.stop()

metrics = json.loads(metrics_path.read_text())
mae = metrics["mae_cycles"]
rmse = metrics["rmse_cycles"]
recall = metrics["critical_recall"] * 100

cards = [
    ("Average prediction error", f"{mae:.1f} cycles", "The forecast is usually off by about this many operating cycles."),
    ("Bigger-miss score", f"{rmse:.1f} cycles", "A cautious error score that gives extra weight to the model's largest mistakes."),
    ("At-risk engines found", f"{recall:.0f}%", "The share of engines near maintenance that received an early warning."),
    ("Unseen test engines", str(metrics["test_engines"]), "Engines kept out of training so the evaluation is fair."),
]
card_html = "".join(
    f'<div class="plain-card"><div class="plain-label">{label}</div>'
    f'<div class="plain-value">{value}</div><div class="plain-help">{help_text}</div></div>'
    for label, value, help_text in cards
)
st.markdown(
    f"""
    <style>
    .plain-grid {{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px;margin:18px 0 28px}}
    .plain-card {{background:white;border:1px solid #e4e8ef;border-radius:14px;padding:18px;min-width:0;box-shadow:0 3px 12px rgba(20,40,70,.06)}}
    .plain-label {{font-size:.88rem;color:#52606d;min-height:2.5em}}
    .plain-value {{font-size:1.8rem;font-weight:750;color:#123b63;white-space:normal;margin:7px 0}}
    .plain-help {{font-size:.82rem;line-height:1.4;color:#687787}}
    @media(max-width:900px) {{.plain-grid {{grid-template-columns:repeat(2,minmax(0,1fr))}}}}
    @media(max-width:520px) {{.plain-grid {{grid-template-columns:1fr}}}}
    </style>
    <div class="plain-grid">{card_html}</div>
    """,
    unsafe_allow_html=True,
)

st.subheader("What this means for a maintenance team")
left, middle, right = st.columns(3)
left.success("PLAN\n\nSchedule inspections before an engine reaches its predicted maintenance window.")
middle.warning("PRIORITIZE\n\nReview engines with 30 or fewer estimated operating cycles remaining.")
right.info("VERIFY\n\nUse the prediction as decision support, not as a replacement for a qualified inspection.")

importance = pd.read_csv(ROOT / "artifacts" / "feature_importance.csv").head(10).copy()
importance["Business-friendly signal"] = importance["feature"].map(FEATURE_NAMES).fillna(importance["feature"])
importance = importance.sort_values("importance")
figure = px.bar(
    importance,
    x="importance",
    y="Business-friendly signal",
    orientation="h",
    title="Signals that influence the forecast most",
    labels={"importance": "Influence on prediction", "Business-friendly signal": ""},
    color="importance",
    color_continuous_scale="Blues",
)
figure.update_layout(coloraxis_showscale=False)
st.plotly_chart(figure, width="stretch")

with st.expander("Plain-language model card"):
    st.markdown(f"""
**Purpose:** Estimate remaining operating cycles so a maintenance team can inspect higher-risk aircraft engines earlier.

**Evaluation:** The model learned from **{metrics['train_engines']} engines** and was evaluated on
**{metrics['test_engines']} different engines** it had never seen during training.

**Typical accuracy:** Its estimates differ from the real maintenance point by about **{mae:.1f} cycles on
average**. The bigger-miss score is **{rmse:.1f} cycles**, which highlights occasional larger errors.

**Safety-focused result:** It identified **{recall:.0f}% of engines** that were within the high-risk
30-cycle window. Missing one of these engines matters more than raising an extra inspection alert.

**Limitations:** This portfolio demonstration uses NASA's simulated turbofan-engine dataset. A real deployment
must be retrained and validated using the airline's own engines, operating conditions, and maintenance records.
""")

st.caption("The original sensor identifiers remain in the model for reproducibility; the dashboard explains them as aircraft-engine condition signals for non-specialist viewers.")
