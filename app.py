from pathlib import Path
import json

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).parent
st.set_page_config(page_title="Aircraft Predictive Maintenance", layout="wide")
st.title("Aircraft Predictive Maintenance - Model Card")
metrics_path = ROOT / "artifacts" / "metrics.json"
if not metrics_path.exists():
    st.error("Run `python train.py` first.")
    st.stop()
metrics = json.loads(metrics_path.read_text())
for col, key in zip(st.columns(4), ["mae_cycles", "rmse_cycles", "critical_recall", "test_engines"]):
    col.metric(key.replace("_", " ").title(), metrics[key])
importance = pd.read_csv(ROOT / "artifacts" / "feature_importance.csv").head(12)
st.plotly_chart(px.bar(importance, x="importance", y="feature", orientation="h", title="Permutation importance"), width="stretch")
st.info("Demo metrics are not resume claims. Retrain with NASA FD001 and report that held-out engine result.")

