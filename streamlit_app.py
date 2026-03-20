"""
streamlit_app.py — Fuel or Fool? Vehicle CO2 Emission Predictor & EU Green Rating
"""

import re
import pickle
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Fuel or Fool? — CO2 Predictor",
    page_icon="⛽",
    layout="wide",
)

# ─────────────────────────────────────────────
# Color palette
# ─────────────────────────────────────────────

RATING_COLORS = {
    "A": "#00C853", "B": "#64DD17", "C": "#FFD600",
    "D": "#FFAB00", "E": "#FF6D00", "F": "#FF3D00", "G": "#D50000",
}
SKODA_GREEN = "#4EA94B"

FUEL_LABELS = {
    "X": "Regular Gas", "Z": "Premium Gas",
    "D": "Diesel", "E": "Ethanol E85", "N": "Natural Gas",
}

# ─────────────────────────────────────────────
# Load artifacts
# ─────────────────────────────────────────────

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

@st.cache_data
def load_feature_names():
    with open("models/feature_names.pkl", "rb") as f:
        return pickle.load(f)

model        = load_model()
scaler       = load_scaler()
encoders     = load_encoders()
stats        = load_stats()
metrics      = load_metrics()
feature_names = load_feature_names()

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def get_eu_rating(co2: float) -> str:
    if co2 <= 100: return "A"
    if co2 <= 120: return "B"
    if co2 <= 140: return "C"
    if co2 <= 160: return "D"
    if co2 <= 200: return "E"
    if co2 <= 250: return "F"
    return "G"

def co2_percentile(co2: float) -> float:
    arr = np.array(stats["all_co2"])
    return float(np.mean(arr < co2) * 100)

def parse_transmission_type(label: str) -> str:
    mapping = {
        "Automatic": "A",
        "Automated Manual": "AM",
        "Auto Select Shift": "AS",
        "CVT": "AV",
        "Manual": "M",
    }
    return mapping.get(label, "A")

def encode(col: str, val: str) -> int:
    le = encoders[col]
    if val in le.classes_:
        return int(le.transform([val])[0])
    return int(le.transform([le.classes_[0]])[0])

def predict_co2(make, vehicle_class, engine_size, cylinders,
                transmission_label, speeds, fuel_label, fuel_city, fuel_hwy):
    fuel_code = fuel_label.split("(")[-1].rstrip(")")
    trans_type = parse_transmission_type(transmission_label)
    ratio = fuel_city / max(fuel_hwy, 0.1)

    x = np.array([[
        engine_size,
        cylinders,
        fuel_city,
        fuel_hwy,
        ratio,
        speeds,
        encode("make", make),
        encode("vehicle_class", vehicle_class),
        encode("transmission_type", trans_type),
        encode("fuel_type", fuel_code),
    ]])
    x_scaled = scaler.transform(x)
    return float(model.predict(x_scaled)[0])

# ─────────────────────────────────────────────
# Session state defaults
# ─────────────────────────────────────────────

DEFAULTS = dict(
    make="VOLKSWAGEN",
    vehicle_class="COMPACT",
    engine_size=2.0,
    cylinders=4,
    transmission="Automatic",
    speeds=6,
    fuel_label="Regular Gasoline (X)",
    fuel_city=10.0,
    fuel_hwy=7.0,
)

for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────

st.markdown("# ⛽ Fuel or Fool?")
st.markdown("### Vehicle CO2 Emission Predictor & EU Green Rating")
st.markdown("Predict CO2 emissions for any car configuration and see how it compares to the market.")
st.divider()

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown(f"## 🚗 Configure Your Car")

    # ── Presets ──
    st.markdown("**Quick Presets**")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🌿 Eco Compact", use_container_width=True):
            st.session_state.update(
                make="VOLKSWAGEN", vehicle_class="COMPACT", engine_size=1.4,
                cylinders=4, transmission="Manual", speeds=6,
                fuel_label="Regular Gasoline (X)", fuel_city=6.5, fuel_hwy=5.0,
            )
        if st.button("🏎️ Sports Car", use_container_width=True):
            st.session_state.update(
                make="BMW", vehicle_class="TWO-SEATER", engine_size=3.0,
                cylinders=6, transmission="Automatic", speeds=8,
                fuel_label="Premium Gasoline (Z)", fuel_city=15.0, fuel_hwy=10.0,
            )
    with col_b:
        if st.button("🏙️ City SUV", use_container_width=True):
            st.session_state.update(
                make="TOYOTA", vehicle_class="SUV - SMALL", engine_size=2.0,
                cylinders=4, transmission="Automatic", speeds=6,
                fuel_label="Regular Gasoline (X)", fuel_city=11.0, fuel_hwy=8.0,
            )
        if st.button("🛻 Big Pickup", use_container_width=True):
            st.session_state.update(
                make="FORD", vehicle_class="PICKUP TRUCK - STANDARD", engine_size=5.3,
                cylinders=8, transmission="Automatic", speeds=6,
                fuel_label="Regular Gasoline (X)", fuel_city=18.0, fuel_hwy=13.0,
            )

    st.divider()

    # ── Inputs ──
    makes = stats["makes"]
    default_make_idx = makes.index("VOLKSWAGEN") if "VOLKSWAGEN" in makes else 0
    make = st.selectbox("Make", makes,
                        index=makes.index(st.session_state.make) if st.session_state.make in makes else default_make_idx)

    vclasses = stats["vehicle_classes"]
    vehicle_class = st.selectbox("Vehicle Class", vclasses,
                                 index=vclasses.index(st.session_state.vehicle_class) if st.session_state.vehicle_class in vclasses else 0)

    engine_size = st.slider("Engine Size (L)", 1.0, 8.0,
                            float(st.session_state.engine_size), step=0.1)

    cylinders = st.selectbox("Cylinders", [3, 4, 5, 6, 8, 10, 12],
                             index=[3, 4, 5, 6, 8, 10, 12].index(st.session_state.cylinders)
                             if st.session_state.cylinders in [3, 4, 5, 6, 8, 10, 12] else 1)

    transmission_options = ["Automatic", "Automated Manual", "Auto Select Shift", "CVT", "Manual"]
    transmission = st.selectbox("Transmission", transmission_options,
                                index=transmission_options.index(st.session_state.transmission)
                                if st.session_state.transmission in transmission_options else 0)

    speeds = st.slider("Transmission Speeds", 4, 10, int(st.session_state.speeds))

    fuel_options = [
        "Regular Gasoline (X)", "Premium Gasoline (Z)", "Diesel (D)",
        "Ethanol E85 (E)", "Natural Gas (N)",
    ]
    fuel_label = st.selectbox("Fuel Type", fuel_options,
                              index=fuel_options.index(st.session_state.fuel_label)
                              if st.session_state.fuel_label in fuel_options else 0)

    fuel_city = st.slider("Fuel Consumption City (L/100 km)", 4.0, 30.0,
                          float(st.session_state.fuel_city), step=0.1)
    fuel_hwy = st.slider("Fuel Consumption Highway (L/100 km)", 4.0, 20.0,
                         float(st.session_state.fuel_hwy), step=0.1)

    # Persist to session state
    st.session_state.update(
        make=make, vehicle_class=vehicle_class, engine_size=engine_size,
        cylinders=cylinders, transmission=transmission, speeds=speeds,
        fuel_label=fuel_label, fuel_city=fuel_city, fuel_hwy=fuel_hwy,
    )

# ─────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────

co2_pred = predict_co2(
    make, vehicle_class, engine_size, cylinders,
    transmission, speeds, fuel_label, fuel_city, fuel_hwy
)
co2_pred = max(96, min(522, co2_pred))
rating = get_eu_rating(co2_pred)
rating_color = RATING_COLORS[rating]
percentile = co2_percentile(co2_pred)
market_median = stats["co2_percentiles"][50]

# ─────────────────────────────────────────────
# Row 1 — Three KPI columns
# ─────────────────────────────────────────────

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("#### 🌍 CO2 Emissions")
    delta = co2_pred - market_median
    delta_str = f"{delta:+.0f} g/km vs market median"
    st.metric(
        label="Predicted Emissions",
        value=f"{co2_pred:.0f} g/km",
        delta=delta_str,
        delta_color="inverse",
    )

with c2:
    st.markdown("#### 🏷️ EU Green Rating")
    badge_html = f"""
    <div style="
        display: inline-block;
        background-color: {rating_color};
        color: {'#000' if rating in ('B','C','D') else '#fff'};
        font-size: 72px;
        font-weight: 900;
        width: 110px;
        height: 110px;
        line-height: 110px;
        text-align: center;
        border-radius: 16px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.25);
        font-family: sans-serif;
    ">{rating}</div>
    <div style="margin-top:8px; font-size:14px; color:#555;">
        {rating}: {
            {'A':'≤ 100','B':'101–120','C':'121–140','D':'141–160',
             'E':'161–200','F':'201–250','G':'251+'}[rating]
        } g/km
    </div>
    """
    st.markdown(badge_html, unsafe_allow_html=True)

with c3:
    st.markdown("#### 📊 Market Percentile")
    cleaner_pct = 100 - percentile
    st.markdown(f"**Your car emits less CO₂ than {cleaner_pct:.0f}% of vehicles**")
    st.progress(cleaner_pct / 100)
    st.caption(f"Market median: {market_median:.0f} g/km · Your car: {co2_pred:.0f} g/km")

st.divider()

# ─────────────────────────────────────────────
# Row 2 — EU-style A–G scale
# ─────────────────────────────────────────────

st.markdown("#### EU Energy Label Scale")

rating_boundaries = [
    ("A", 0,   100),
    ("B", 100, 120),
    ("C", 120, 140),
    ("D", 140, 160),
    ("E", 160, 200),
    ("F", 200, 250),
    ("G", 250, 380),
]

fig_scale = go.Figure()

# Stacked horizontal segments
for ltr, lo, hi in rating_boundaries:
    width = hi - lo
    fig_scale.add_trace(go.Bar(
        x=[width], y=["Rating"],
        orientation="h",
        marker_color=RATING_COLORS[ltr],
        name=ltr,
        hovertemplate=f"<b>{ltr}</b>: {lo}–{hi} g/km<extra></extra>",
        text=ltr,
        textposition="inside",
        insidetextanchor="middle",
        textfont=dict(size=16, color="white" if ltr in ("A", "E", "F", "G") else "black"),
    ))

# Marker line for predicted value
fig_scale.add_vline(
    x=min(co2_pred, 380) - rating_boundaries[0][1],
    line_width=3,
    line_dash="solid",
    line_color="#1E1E1E",
    annotation_text=f" {co2_pred:.0f} g/km",
    annotation_position="top",
    annotation_font_size=13,
)

fig_scale.update_layout(
    barmode="stack",
    height=100,
    margin=dict(l=0, r=10, t=30, b=10),
    showlegend=False,
    xaxis=dict(visible=False),
    yaxis=dict(visible=False),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig_scale, use_container_width=True)

st.divider()

# ─────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["📊 How You Compare", "🔍 What Drives Emissions", "🤖 Model Info"])

# ── Tab 1: How You Compare ──────────────────

with tab1:
    col_l, col_r = st.columns(2)

    with col_l:
        # CO2 distribution histogram
        st.markdown("##### CO2 Distribution Across All Vehicles")
        fig_hist = px.histogram(
            x=stats["all_co2"], nbins=60,
            labels={"x": "CO2 Emissions (g/km)", "y": "Count"},
            color_discrete_sequence=[SKODA_GREEN],
            opacity=0.8,
        )
        fig_hist.add_vline(
            x=co2_pred, line_dash="dash", line_color="red", line_width=2,
            annotation_text=f"Your car: {co2_pred:.0f}", annotation_position="top right",
            annotation_font_color="red",
        )
        fig_hist.update_layout(showlegend=False, height=320,
                               margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_hist, use_container_width=True)

        # CO2 by fuel type
        st.markdown("##### Average CO2 by Fuel Type")
        fuel_co2 = {FUEL_LABELS.get(k, k): v for k, v in stats["co2_by_fuel_type"].items()}
        fig_fuel = px.bar(
            x=list(fuel_co2.values()), y=list(fuel_co2.keys()),
            orientation="h", color=list(fuel_co2.keys()),
            color_discrete_sequence=px.colors.qualitative.Set2,
            labels={"x": "Avg CO2 (g/km)", "y": ""},
        )
        fig_fuel.update_layout(showlegend=False, height=260,
                               margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_fuel, use_container_width=True)

    with col_r:
        # CO2 by vehicle class
        st.markdown("##### Average CO2 by Vehicle Class")
        vc_co2 = dict(sorted(stats["co2_by_vehicle_class"].items(), key=lambda x: x[1]))
        bar_colors = [RATING_COLORS["A"] if k == vehicle_class else SKODA_GREEN
                      for k in vc_co2.keys()]
        fig_vc = go.Figure(go.Bar(
            x=list(vc_co2.values()), y=list(vc_co2.keys()),
            orientation="h", marker_color=bar_colors,
            hovertemplate="%{y}: %{x:.0f} g/km<extra></extra>",
        ))
        fig_vc.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0),
                             xaxis_title="Avg CO2 (g/km)")
        st.plotly_chart(fig_vc, use_container_width=True)

        # Top-20 makes
        st.markdown("##### Average CO2 by Make (Top 20)")
        make_co2 = dict(sorted(stats["co2_by_make"].items(), key=lambda x: x[1])[:20])
        bar_colors_make = [RATING_COLORS["A"] if k == make else SKODA_GREEN
                           for k in make_co2.keys()]
        fig_make = go.Figure(go.Bar(
            x=list(make_co2.values()), y=list(make_co2.keys()),
            orientation="h", marker_color=bar_colors_make,
            hovertemplate="%{y}: %{x:.0f} g/km<extra></extra>",
        ))
        fig_make.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0),
                               xaxis_title="Avg CO2 (g/km)")
        st.plotly_chart(fig_make, use_container_width=True)

# ── Tab 2: What Drives Emissions ────────────

with tab2:
    col_l, col_r = st.columns(2)
    df_plots = stats["df_for_plots"]

    import pandas as pd
    df_vis = pd.DataFrame(df_plots)

    with col_l:
        # Feature importance
        st.markdown("##### Feature Importance")
        fi = metrics.get("feature_importance", {})
        fi_sorted = dict(sorted(fi.items(), key=lambda x: x[1]))
        fig_fi = go.Figure(go.Bar(
            x=list(fi_sorted.values()), y=list(fi_sorted.keys()),
            orientation="h", marker_color=SKODA_GREEN,
        ))
        fig_fi.update_layout(height=340, margin=dict(l=0, r=0, t=10, b=0),
                             xaxis_title="Importance")
        st.plotly_chart(fig_fi, use_container_width=True)

        # Cylinders vs CO2
        st.markdown("##### Cylinders vs CO2")
        df_vis["cylinders_str"] = df_vis["cylinders"].astype(str) + " cyl"
        fig_cyl = px.box(
            df_vis, x="cylinders", y="co2_emissions",
            color="cylinders_str",
            color_discrete_sequence=px.colors.sequential.Viridis,
            labels={"cylinders": "Cylinders", "co2_emissions": "CO2 (g/km)"},
        )
        fig_cyl.add_scatter(
            x=[cylinders], y=[co2_pred],
            mode="markers", marker=dict(color="red", size=14, symbol="star"),
            name="Your car",
        )
        fig_cyl.update_layout(showlegend=False, height=300,
                              margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_cyl, use_container_width=True)

    with col_r:
        # Engine size vs CO2
        st.markdown("##### Engine Size vs CO2")
        df_vis["fuel_label"] = df_vis["fuel_type"].map(FUEL_LABELS)
        fig_eng = px.scatter(
            df_vis.sample(min(2000, len(df_vis)), random_state=1),
            x="engine_size", y="co2_emissions",
            color="fuel_label", opacity=0.45,
            labels={"engine_size": "Engine Size (L)", "co2_emissions": "CO2 (g/km)",
                    "fuel_label": "Fuel Type"},
            color_discrete_sequence=px.colors.qualitative.Safe,
        )
        fig_eng.add_scatter(
            x=[engine_size], y=[co2_pred],
            mode="markers", marker=dict(color="red", size=16, symbol="star"),
            name="Your car",
        )
        fig_eng.update_layout(height=340, margin=dict(l=0, r=0, t=10, b=0),
                              legend=dict(orientation="h", yanchor="bottom", y=1.0))
        st.plotly_chart(fig_eng, use_container_width=True)

        # Avg CO2 by fuel type table
        st.markdown("##### CO2 by Fuel Type")
        fuel_table = pd.DataFrame([
            {"Fuel Type": FUEL_LABELS.get(k, k), "Avg CO2 (g/km)": f"{v:.1f}"}
            for k, v in sorted(stats["co2_by_fuel_type"].items(), key=lambda x: x[1])
        ])
        st.dataframe(fuel_table, use_container_width=True, hide_index=True)

# ── Tab 3: Model Info ────────────────────────

with tab3:
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("##### Model Comparison")
        best_name = metrics["best_model_name"]
        model_rows = []
        for name in ["Ridge Regression", "Random Forest", "Gradient Boosting"]:
            m = metrics.get(name, {})
            model_rows.append({
                "Model": f"⭐ {name}" if name == best_name else name,
                "MAE": f"{m.get('MAE', 0):.2f}",
                "RMSE": f"{m.get('RMSE', 0):.2f}",
                "R²": f"{m.get('R2', 0):.4f}",
                "MAPE%": f"{m.get('MAPE', 0):.2f}%",
            })
        import pandas as pd
        st.dataframe(pd.DataFrame(model_rows), use_container_width=True, hide_index=True)

        st.markdown("##### About the Project")
        st.markdown("""
**Dataset**: CO2 Emissions by Vehicles (~7 400 records, Canadian market)

**Target variable**: CO2 Emissions (g/km)

**Key insight**: `fuel_comb` was intentionally **excluded** from features — CO2 is
almost a mathematical function of combined fuel consumption × emission factor,
which would create data leakage. Instead the model uses city/highway consumption
separately alongside engine specs and car attributes.

**EU Green Rating** is based on the EU Energy Label thresholds (g CO₂/km):

| Rating | Range |
|--------|-------|
| A | ≤ 100 g/km |
| B | 101–120 g/km |
| C | 121–140 g/km |
| D | 141–160 g/km |
| E | 161–200 g/km |
| F | 201–250 g/km |
| G | > 250 g/km |

**Links**: [GitHub](#) · [Kaggle Dataset](#) · [Live Demo](#)
        """)

    with col_r:
        st.markdown("##### Predicted vs Actual (Test Set)")
        y_test = metrics.get("y_test", [])
        y_pred = metrics.get("y_pred", [])
        if len(y_test) > 0:
            import pandas as pd
            fig_pva = px.scatter(
                x=y_test, y=y_pred, opacity=0.4,
                labels={"x": "Actual CO2 (g/km)", "y": "Predicted CO2 (g/km)"},
                color_discrete_sequence=[SKODA_GREEN],
            )
            lo, hi = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
            fig_pva.add_scatter(
                x=[lo, hi], y=[lo, hi], mode="lines",
                line=dict(color="red", dash="dash"), name="Perfect prediction",
            )
            fig_pva.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_pva, use_container_width=True)
        else:
            st.info("Test predictions not available.")
