"""
streamlit_app.py  —  Fuel or Fool? (v0.1 basic)

First working version: sidebar inputs → predict → show CO2 + basic charts.
EU badge and A-G scale coming in next commit.
"""

import pickle
import re
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(
    page_title="Fuel or Fool? — CO2 Predictor",
    page_icon="⛽",
    layout="wide",
)

RATING_COLORS = {
    "A": "#00C853", "B": "#64DD17", "C": "#FFD600",
    "D": "#FFAB00", "E": "#FF6D00", "F": "#FF3D00", "G": "#D50000",
}
SKODA_GREEN = "#4EA94B"
FUEL_LABELS = {
    "X": "Regular Gas", "Z": "Premium Gas",
    "D": "Diesel", "E": "Ethanol E85", "N": "Natural Gas",
}


@st.cache_resource
def load_model():
    with open("models/co2_model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_scaler():
    with open("models/scaler.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_encoders():
    with open("models/label_encoders.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_stats():
    with open("models/data_stats.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_metrics():
    with open("models/model_metrics.pkl", "rb") as f:
        return pickle.load(f)


model    = load_model()
scaler   = load_scaler()
encoders = load_encoders()
stats    = load_stats()
metrics  = load_metrics()


def get_eu_rating(co2: float) -> str:
    if co2 <= 100:  return "A"
    if co2 <= 120:  return "B"
    if co2 <= 140:  return "C"
    if co2 <= 160:  return "D"
    if co2 <= 200:  return "E"
    if co2 <= 250:  return "F"
    return "G"

def co2_percentile(co2: float) -> float:
    arr = np.array(stats["all_co2"])
    return float(np.mean(arr < co2) * 100)

def encode(col: str, val: str) -> int:
    le = encoders[col]
    val = val if val in le.classes_ else le.classes_[0]
    return int(le.transform([val])[0])

def parse_trans_type(label: str) -> str:
    return {"Automatic": "A", "Automated Manual": "AM",
            "Auto Select Shift": "AS", "CVT": "AV", "Manual": "M"}.get(label, "A")

def predict(make, vehicle_class, engine_size, cylinders,
            trans_label, speeds, fuel_label, fuel_city, fuel_hwy):
    fuel_code  = fuel_label.split("(")[-1].rstrip(")")
    trans_type = parse_trans_type(trans_label)
    ratio      = fuel_city / max(fuel_hwy, 0.1)
    x = np.array([[engine_size, cylinders, fuel_city, fuel_hwy, ratio, speeds,
                   encode("make", make), encode("vehicle_class", vehicle_class),
                   encode("transmission_type", trans_type),
                   encode("fuel_type", fuel_code)]])
    return float(model.predict(scaler.transform(x))[0])


# ── Header ──────────────────────────────────────────────────────────────────

st.markdown("# ⛽ Fuel or Fool?")
st.markdown("### Vehicle CO2 Emission Predictor & EU Green Rating")
st.markdown("Predict CO2 emissions for any car configuration and see how it compares to the market.")
st.divider()


# ── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🚗 Configure Your Car")

    makes = stats["makes"]
    make = st.selectbox("Make", makes,
                        index=makes.index("VOLKSWAGEN") if "VOLKSWAGEN" in makes else 0)

    vclasses = stats["vehicle_classes"]
    vehicle_class = st.selectbox("Vehicle Class", vclasses,
                                 index=vclasses.index("COMPACT") if "COMPACT" in vclasses else 0)

    engine_size = st.slider("Engine Size (L)", 1.0, 8.0, 2.0, step=0.1)
    cylinders   = st.selectbox("Cylinders", [3, 4, 5, 6, 8, 10, 12], index=1)

    trans_options = ["Automatic", "Automated Manual", "Auto Select Shift", "CVT", "Manual"]
    transmission  = st.selectbox("Transmission", trans_options)
    speeds        = st.slider("Transmission Speeds", 4, 10, 6)

    fuel_options = ["Regular Gasoline (X)", "Premium Gasoline (Z)", "Diesel (D)",
                    "Ethanol E85 (E)", "Natural Gas (N)"]
    fuel_label   = st.selectbox("Fuel Type", fuel_options)

    fuel_city = st.slider("Fuel Consumption City (L/100 km)", 4.0, 30.0, 10.0, step=0.1)
    fuel_hwy  = st.slider("Fuel Consumption Hwy (L/100 km)",  4.0, 20.0,  7.0, step=0.1)


# ── Prediction ───────────────────────────────────────────────────────────────

co2_pred = float(np.clip(
    predict(make, vehicle_class, engine_size, cylinders,
            transmission, speeds, fuel_label, fuel_city, fuel_hwy),
    96, 522
))
rating        = get_eu_rating(co2_pred)
rating_color  = RATING_COLORS[rating]
percentile    = co2_percentile(co2_pred)
market_median = stats["co2_percentiles"][50]


# ── KPI row ──────────────────────────────────────────────────────────────────

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("#### 🌍 CO2 Emissions")
    st.metric("Predicted", f"{co2_pred:.0f} g/km",
              delta=f"{co2_pred - market_median:+.0f} g/km vs median",
              delta_color="inverse")

with c2:
    st.markdown("#### 🏷️ EU Green Rating")
    st.markdown(
        f'<div style="display:inline-block;background:{rating_color};'
        f'color:{"#000" if rating in "BCD" else "#fff"};'
        f'font-size:72px;font-weight:900;width:110px;height:110px;'
        f'line-height:110px;text-align:center;border-radius:16px;'
        f'box-shadow:0 4px 12px rgba(0,0,0,0.25)">{rating}</div>'
        f'<div style="margin-top:8px;font-size:14px;color:#555">'
        f'{rating}: {{"A":"≤100","B":"101–120","C":"121–140","D":"141–160",'
        f'"E":"161–200","F":"201–250","G":"251+"}}.get("{rating}","") g/km</div>',
        unsafe_allow_html=True,
    )

with c3:
    st.markdown("#### 📊 Market Percentile")
    cleaner = 100 - percentile
    st.markdown(f"**Cleaner than {cleaner:.0f}% of vehicles**")
    st.progress(cleaner / 100)
    st.caption(f"Market median: {market_median:.0f} g/km")

st.divider()


# ── Basic charts ─────────────────────────────────────────────────────────────

st.markdown("#### CO2 Distribution")
fig = px.histogram(x=stats["all_co2"], nbins=60, color_discrete_sequence=[SKODA_GREEN],
                   labels={"x": "CO2 (g/km)", "y": "Count"})
fig.add_vline(x=co2_pred, line_dash="dash", line_color="red",
              annotation_text=f"Your car: {co2_pred:.0f}", annotation_position="top right",
              annotation_font_color="red")
fig.update_layout(showlegend=False, height=300, margin=dict(l=0,r=0,t=10,b=0))
st.plotly_chart(fig, use_container_width=True)
