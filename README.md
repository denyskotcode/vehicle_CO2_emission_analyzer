# ⛽ Fuel or Fool?
### Vehicle CO2 Emission Predictor & EU Green Rating System

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-5.x-3F4F75?style=flat&logo=plotly&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-4EA94B?style=flat)

> **Can a machine tell you if your car is eco-friendly or just a fool for the environment?**  
> Enter any car configuration — get an instant CO2 prediction and an EU-style green rating from **A** to **G**.

---

## 🎯 What this project does

I built a Streamlit web app that predicts CO2 emissions (g/km) for any car configuration using machine learning. The model is trained on ~7,400 vehicles from the Canadian market and outputs:

- A **predicted CO2 value** in g/km
- An **EU Energy Label rating** (A–G) with a color-coded badge
- A **market percentile** — where does your car stand vs 7,400 others?
- **Interactive charts** to explore what actually drives emissions

---

## 📸 App Preview

```
┌─────────────────────────────────────────────────────┐
│  ⛽ Fuel or Fool?                                    │
│  Vehicle CO2 Emission Predictor & EU Green Rating   │
├────────────────┬──────────────┬─────────────────────┤
│ 🌍 187 g/km   │  🏷️  [  E  ] │ 📊 ████████░░  68%  │
│ +35 vs median │  161-200 g/km│ cleaner than 32%    │
├────────────────┴──────────────┴─────────────────────┤
│  [A]▓▓[B]▓▓▓[C]▓▓▓[D]▓▓▓[E]▓▓▓▓[F]▓▓▓▓[G]▓▓▓▓▓   │
│                          ↑ 187 g/km                 │
└─────────────────────────────────────────────────────┘
```

---

## ✨ Features

- **🔮 CO2 Prediction** — from engine size, fuel consumption, transmission, make, and class
- **🏷️ EU Green Rating A–G** — color-coded badge matching real EU Energy Label thresholds
- **📊 Market Comparison** — percentile rank vs 7,400 vehicles + delta from market median
- **📉 Visual A–G Scale** — horizontal Plotly bar with a live marker for your car
- **🚗 Quick Presets** — Eco Compact, City SUV, Sports Car, Big Pickup
- **🔍 Feature Importance** — see which specs actually drive emissions
- **📈 Interactive EDA** — distributions, box plots, scatter plots, correlation heatmap

---

## 🧠 The ML Part

### Why not just use fuel consumption directly?

CO2 is physically calculated from combined fuel consumption:

```
CO2 (g/km) = fuel_comb (L/100km) × emission_factor (g/L) / 100
```

Using `fuel_comb` as a feature would be **data leakage** — the model would just learn this formula and score R²≈1.0 without learning anything useful. Instead, I use `fuel_city` and `fuel_hwy` **separately**, alongside engine specs and car attributes — forcing the model to understand the car, not just the formula.

### Feature set (10 features)

| Feature | Description |
|---------|-------------|
| `engine_size` | Engine displacement in litres |
| `cylinders` | Number of cylinders |
| `fuel_city` | City fuel consumption (L/100 km) |
| `fuel_hwy` | Highway fuel consumption (L/100 km) |
| `fuel_city_hwy_ratio` | City/highway ratio (urban penalty) |
| `transmission_speeds` | Number of gears |
| `make_encoded` | Car brand (label encoded) |
| `vehicle_class_encoded` | Body class (label encoded) |
| `transmission_type_encoded` | Auto / Manual / CVT / etc. |
| `fuel_type_encoded` | Gasoline / Diesel / Ethanol / CNG |

### Model comparison

| Model | MAE | RMSE | R² | MAPE |
|-------|-----|------|----|------|
| Ridge Regression | ~16 g/km | ~21 g/km | 0.81 | 6.9% |
| Random Forest | ~6 g/km | ~7 g/km | 0.979 | 2.4% |
| **Gradient Boosting ⭐** | **~5 g/km** | **~7 g/km** | **0.981** | **2.3%** |

Gradient Boosting wins — 200 trees, each correcting the errors of the previous one.

---

## 🟢 EU Green Rating Scale

| Rating | CO2 Emissions | Examples |
|--------|--------------|---------|
| 🟢 **A** | ≤ 100 g/km | Small EVs, mild hybrids |
| 🟢 **B** | 101 – 120 g/km | Efficient compacts |
| 🟡 **C** | 121 – 140 g/km | Average compact cars |
| 🟡 **D** | 141 – 160 g/km | Mid-size sedans |
| 🟠 **E** | 161 – 200 g/km | SUVs, larger cars |
| 🔴 **F** | 201 – 250 g/km | Performance cars |
| 🔴 **G** | > 250 g/km | Trucks, supercars |

---

## 🚀 Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/denyskotcode/vehicle_CO2_emission_analyzer.git
cd vehicle_CO2_emission_analyzer

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train the model (generates models/*.pkl)
python train_model.py

# 5. Run the app
streamlit run streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501) — the app starts in seconds.

---

## 📁 Project Structure

```
fuel-or-fool/
│
├── 📊 data/
│   └── co2_emissions.csv          # ~7,400 vehicle records
│
├── 📓 notebooks/
│   └── co2_eda.ipynb              # Full EDA: distributions, correlations,
│                                  # model training, feature importance
│
├── 🤖 models/                     # Auto-generated by train_model.py
│   ├── co2_model.pkl              # Best model (Gradient Boosting)
│   ├── scaler.pkl                 # StandardScaler
│   ├── label_encoders.pkl         # LabelEncoders for categoricals
│   ├── feature_names.pkl          # Ordered feature list
│   ├── model_metrics.pkl          # All metrics + test predictions
│   └── data_stats.pkl             # Pre-computed stats for visualizations
│
├── ⚙️  .streamlit/
│   └── config.toml                # Škoda green theme
│
├── 🖥️  streamlit_app.py           # Main application
├── 🏋️  train_model.py             # Data generation + model training
├── 📋 requirements.txt
├── 🙈 .gitignore
└── 📖 README.md
```

---

## 🛠️ Tech Stack

| Layer | Tool | Why |
|-------|------|-----|
| App framework | Streamlit | Fast ML app development in pure Python |
| ML | scikit-learn | Ridge, Random Forest, Gradient Boosting |
| Visualizations | Plotly | Interactive charts, no JS needed |
| Data | pandas + numpy | Data manipulation and feature engineering |
| Serialization | pickle | Save/load model artifacts |

---

## 📊 Dataset

**CO2 Emission by Vehicles** — Canadian Government open data, hosted on Kaggle.  
~7,400 records across 40+ car brands, 15 vehicle classes, 5 fuel types.

> Source: [Kaggle — debajyotipodder/co2-emission-by-vehicles](https://www.kaggle.com/datasets/debajyotipodder/co2-emission-by-vehicles)

If the Kaggle dataset is unavailable, `train_model.py` automatically generates a realistic synthetic dataset with matching statistical properties.

---

## 💡 Key Learnings

Building this project taught me:

1. **Data leakage is subtle** — `fuel_comb` looks like a useful feature until you realize CO2 is literally computed from it. R²=1.0 is a red flag, not a win.
2. **Feature engineering matters** — `fuel_city_hwy_ratio` (city/highway ratio) captures urban driving penalty that neither value alone expresses.
3. **Gradient Boosting vs Random Forest** — both score similarly here (R²≈0.98), but GBR edges out RF because sequential correction handles the remaining nonlinear variance better.
4. **Streamlit session_state** — essential for presets; without it, sidebar sliders reset on every interaction.
5. **Pre-compute, don't compute live** — caching stats in `data_stats.pkl` makes the app feel instant.

---

## 🔗 Links

- 📦 [Dataset on Kaggle](https://www.kaggle.com/datasets/debajyotipodder/co2-emission-by-vehicles)
- 👤 [GitHub](https://github.com/denyskotcode/vehicle_CO2_emission_analyzer)

---

*Built as a portfolio project · March 2026*  
*Inspired by the EU Energy Label system for passenger cars*


---

## License

MIT © [denyskotcode](https://github.com/denyskotcode) — see [LICENSE](LICENSE) for details.