"""Cycle replay page.

    streamlit run app/replay.py

Pick a demo cycle, see its signals against normal reference statistics, the
anomaly score vs. the frozen threshold, and the reference deviation hints.
The known experimental label is shown separately and never enters prediction.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robot_ad.config import MODELS_DIR, NORMAL_SETTING_NAMES, PROCESSED_DIR  # noqa: E402
from robot_ad.predict import Predictor  # noqa: E402
from robot_ad.schema import InputError, cycle_to_payload  # noqa: E402

BUNDLE = os.environ.get("ROBOT_AD_BUNDLE", str(MODELS_DIR / "iforest__full_18.joblib"))

st.set_page_config(page_title="Robot cycle replay", layout="wide")


@st.cache_resource
def load_predictor() -> Predictor:
    return Predictor.from_path(BUNDLE)


@st.cache_data
def load_demo() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / "demo_cycles.parquet")


pred = load_predictor()
demo = load_demo()
cases = {}
try:
    err = json.loads((ROOT / "reports" / "error_analysis.json").read_text(encoding="utf-8"))
    cases = {c["sample"]: c["kind"] for c in err["cases"]}
except FileNotFoundError:
    pass

st.title("Robot pick-and-place cycle — anomaly replay")
st.caption(
    f"Model `{pred.model_version}` · schema `{pred.schema_version}` · threshold {pred.threshold:.4f} "
    "(95 % quantile of calibration normal cycles; alarm if score > threshold)"
)

meta = demo.groupby("sample").first()[["anomaly", "category_name", "setting"]]


def label(s: int) -> str:
    tag = f" · {cases[s]}" if s in cases else ""
    return f"sample {s}{tag}"


sid = st.sidebar.selectbox("Cycle", meta.index.tolist(), format_func=label)
groups = st.sidebar.multiselect(
    "Signal groups", ["motor_iq", "joint_velocity", "torque_sensor_a"], default=["motor_iq", "torque_sensor_a"]
)

cyc = demo[demo["sample"] == sid].sort_values("time")
payload = cycle_to_payload(cyc["time"].to_numpy(), cyc[pred.channels].to_numpy(), pred.channels, pred.schema_version)
try:
    res = pred.predict_payload(payload)  # labels are not part of the payload
except InputError as e:
    st.error(f"Input rejected: {e.code} — {e.message}")
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Model decision", "ANOMALY" if res["is_anomaly"] else "normal")
c2.metric(
    "Anomaly score",
    f"{res['anomaly_score']:.4f}",
    delta=f"{res['anomaly_score'] - res['threshold']:+.4f} vs threshold",
    delta_color="inverse",
)
c3.metric("Points / duration", f"{res['n_points']} / {cyc['time'].iloc[-1]:.2f} s")

with st.expander("Known experimental label (reference only, not used by the model)", expanded=True):
    row = meta.loc[sid]
    truth = "anomaly" if row["anomaly"] else "normal"
    setting = NORMAL_SETTING_NAMES.get(int(row["setting"]), "anomaly experiment")
    agree = (truth == "anomaly") == res["is_anomaly"]
    st.write(
        f"Label: **{truth}** · category `{row['category_name']}` · setting `{setting}` · "
        f"model {'agrees' if agree else '**disagrees**'}"
    )

st.subheader("Reference deviation hints")
st.caption(res["reference_deviations_note"])
st.table(pd.DataFrame(res["reference_deviations"]))

st.subheader("Signals")
dev = pred.bundle["deviation"]
names = pred.bundle["feature_names"]
for g in groups:
    fig, axes = plt.subplots(1, 6, figsize=(18, 2.6), sharex=True)
    for i, ax in enumerate(axes, start=1):
        ch = f"{g}_{i}"
        ax.plot(cyc["time"], cyc[ch], lw=0.8, color="C3")
        j = names.index(f"{ch}__mean")
        m, s = dev.mean_[j], dev.std_[j]
        ax.axhspan(m - 2 * s, m + 2 * s, color="0.85", label="fit_normal mean ± 2 sd of cycle mean")
        ax.set_title(ch, fontsize=9)
        ax.tick_params(labelsize=7)
    axes[0].legend(fontsize=6, loc="lower left")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
