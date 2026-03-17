"""
train_model.py
--------------
Generate (or load) the CO2 emissions dataset, engineer features,
train three regression models and save the best one.

Dataset: CO2 Emission by Vehicles (Canadian market, ~7 400 rows)
https://www.kaggle.com/datasets/debajyotipodder/co2-emission-by-vehicles

NOTE: fuel_comb is intentionally excluded from features.
      CO2 (g/km) = fuel_comb * emission_factor is almost a mathematical
      identity, so including it would be data leakage (R^2 -> 1.0).
      We use fuel_city and fuel_hwy separately instead.
"""

import os
import re
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ── 0. Synthetic dataset (fallback when Kaggle CSV is not available) ────────

def generate_synthetic_dataset(n=7400, seed=42):
    """
    Build a realistic synthetic CO2 dataset.
    Correlations mirror the real Kaggle dataset:
      engine_size <-> CO2  ~0.72
      cylinders   <-> CO2  ~0.72
      fuel_city   <-> CO2  ~0.97
    """
    rng = np.random.default_rng(seed)

    makes = [
        "ACURA", "ALFA ROMEO", "ASTON MARTIN", "AUDI", "BENTLEY", "BMW",
        "BUICK", "CADILLAC", "CHEVROLET", "CHRYSLER", "DODGE", "FIAT",
        "FORD", "GMC", "GENESIS", "HONDA", "HYUNDAI", "INFINITI",
        "JAGUAR", "JEEP", "KIA", "LAMBORGHINI", "LAND ROVER", "LEXUS",
        "LINCOLN", "MASERATI", "MAZDA", "MERCEDES-BENZ", "MINI", "MITSUBISHI",
        "NISSAN", "PORSCHE", "RAM", "ROLLS-ROYCE", "SMART", "SUBARU",
        "TOYOTA", "VOLKSWAGEN", "VOLVO", "ACURA",
    ]

    vehicle_classes = [
        "COMPACT", "SUV - SMALL", "MID-SIZE", "SUV - STANDARD", "SUBCOMPACT",
        "FULL-SIZE", "PICKUP TRUCK - STANDARD", "MINICOMPACT", "TWO-SEATER",
        "STATION WAGON - SMALL", "STATION WAGON - MID-SIZE", "VAN - CARGO",
        "VAN - PASSENGER", "PICKUP TRUCK - SMALL", "SPECIAL PURPOSE VEHICLE",
    ]

    # heavier / larger classes get a bigger engine on average
    class_engine_mult = {
        "MINICOMPACT": 0.75, "SUBCOMPACT": 0.85, "COMPACT": 0.90,
        "STATION WAGON - SMALL": 0.90, "TWO-SEATER": 1.10,
        "MID-SIZE": 1.00, "STATION WAGON - MID-SIZE": 1.00,
        "FULL-SIZE": 1.10, "SUV - SMALL": 1.05, "SUV - STANDARD": 1.20,
        "SPECIAL PURPOSE VEHICLE": 1.15, "PICKUP TRUCK - SMALL": 1.15,
        "PICKUP TRUCK - STANDARD": 1.40, "VAN - CARGO": 1.25,
        "VAN - PASSENGER": 1.25,
    }

    transmissions = [
        "A5", "A6", "A7", "A8", "A9", "A10",
        "AM6", "AM7", "AS5", "AS6", "AS7", "AS8", "AS9", "AS10",
        "AV", "AV7", "AV8", "AV10", "M5", "M6", "M7", "A4", "A8",
    ]

    fuel_types   = ["X",    "Z",    "D",    "E",    "N"]
    fuel_weights = [0.55,  0.25,  0.12,  0.05,  0.03]

    # CO2 (g/km) = fuel_comb (L/100 km) * factor / 100
    # gasoline ~2 289 g/L, diesel ~2 640 g/L, ethanol ~1 520 g/L, CNG ~1 900 g/L
    fuel_co2_factor = {"X": 22.89, "Z": 22.89, "D": 26.40, "E": 15.20, "N": 19.00}

    rows = []
    for _ in range(n):
        vc   = rng.choice(vehicle_classes)
        mult = class_engine_mult[vc]

        engine_size = float(np.clip(rng.normal(2.2 * mult, 0.7), 1.0, 8.4))
        engine_size = round(engine_size * 2) / 2          # round to 0.5 L steps

        if engine_size <= 1.5:
            cylinders = rng.choice([3, 4], p=[0.4, 0.6])
        elif engine_size <= 2.5:
            cylinders = rng.choice([4, 6], p=[0.75, 0.25])
        elif engine_size <= 4.0:
            cylinders = rng.choice([4, 6, 8], p=[0.20, 0.55, 0.25])
        elif engine_size <= 6.0:
            cylinders = rng.choice([6, 8, 10], p=[0.30, 0.55, 0.15])
        else:
            cylinders = rng.choice([8, 10, 12, 16], p=[0.40, 0.20, 0.35, 0.05])

        transmission = rng.choice(transmissions)
        fuel_type    = rng.choice(fuel_types, p=fuel_weights)
        make         = rng.choice(makes)

        # base consumption rises with engine size and class weight
        base_city = 5.0 + engine_size * 1.8 + cylinders * 0.3 + mult * 1.5
        base_city = float(np.clip(base_city + rng.normal(0, 0.8), 4.0, 30.0))

        city_hwy_diff = rng.uniform(1.5, 5.0)
        base_hwy = float(np.clip(base_city - city_hwy_diff, 4.0, 20.0))

        # diesel burns less per km (efficiency bonus)
        if fuel_type == "D":
            base_city *= 0.88
            base_hwy  *= 0.88
        elif fuel_type == "E":          # ethanol has lower energy density
            base_city *= 1.25
            base_hwy  *= 1.25

        fuel_city    = round(base_city, 1)
        fuel_hwy     = round(base_hwy,  1)
        fuel_comb    = round(0.55 * fuel_city + 0.45 * fuel_hwy, 1)
        fuel_comb_mpg = round(235.21 / fuel_comb, 0)

        # CO2 = comb * factor + real-world spread (weight, aero, tyre rolling resistance)
        co2  = fuel_comb * fuel_co2_factor[fuel_type]
        co2 *= rng.uniform(0.97, 1.03)   # ±3 % multiplicative
        co2 += rng.normal(0, 4)          # small additive noise
        co2  = int(np.clip(co2, 96, 522))

        rows.append({
            "Make":                              make,
            "Model":                             f"MODEL-{rng.integers(100, 999)}",
            "Vehicle Class":                     vc,
            "Engine Size(L)":                    engine_size,
            "Cylinders":                         cylinders,
            "Transmission":                      transmission,
            "Fuel Type":                         fuel_type,
            "Fuel Consumption City (L/100 km)":  fuel_city,
            "Fuel Consumption Hwy (L/100 km)":   fuel_hwy,
            "Fuel Consumption Comb (L/100 km)":  fuel_comb,
            "Fuel Consumption Comb (mpg)":       int(fuel_comb_mpg),
            "CO2 Emissions(g/km)":               co2,
        })

    return pd.DataFrame(rows)


# ── 1. Load or generate data ────────────────────────────────────────────────

DATA_PATH = "data/co2_emissions.csv"

if os.path.exists(DATA_PATH):
    print(f"Loading dataset from {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)
else:
    print("Dataset not found — generating synthetic data (7 400 rows) ...")
    os.makedirs("data", exist_ok=True)
    df = generate_synthetic_dataset(7400)
    df.to_csv(DATA_PATH, index=False)
    print(f"Saved → {DATA_PATH}")

# rename Kaggle file if present
real_path = "data/CO2 Emissions_Canada.csv"
if os.path.exists(real_path) and not os.path.exists(DATA_PATH):
    os.rename(real_path, DATA_PATH)

print(f"Raw shape: {df.shape}")


# ── 2. Clean & rename columns ───────────────────────────────────────────────

col_map = {
    "Make":                              "make",
    "Model":                             "model",
    "Vehicle Class":                     "vehicle_class",
    "Engine Size(L)":                    "engine_size",
    "Cylinders":                         "cylinders",
    "Transmission":                      "transmission",
    "Fuel Type":                         "fuel_type",
    "Fuel Consumption City (L/100 km)":  "fuel_city",
    "Fuel Consumption Hwy (L/100 km)":   "fuel_hwy",
    "Fuel Consumption Comb (L/100 km)":  "fuel_comb",
    "Fuel Consumption Comb (mpg)":       "fuel_comb_mpg",
    "CO2 Emissions(g/km)":              "co2_emissions",
}

df = df.rename(columns=col_map).drop_duplicates().dropna()
print(f"After cleaning: {df.shape}")


# ── 3. Feature engineering ──────────────────────────────────────────────────

def transmission_type(t: str) -> str:
    t = t.strip()
    for prefix in ("AM", "AS", "AV"):
        if t.startswith(prefix):
            return prefix
    return t[0] if t else "A"

def transmission_speeds(t: str) -> int:
    digits = re.findall(r"\d+", t)
    return int(digits[0]) if digits else 6

df["transmission_type"]   = df["transmission"].apply(transmission_type)
df["transmission_speeds"] = df["transmission"].apply(transmission_speeds)
# city/hwy ratio captures how "urban" the driving profile is
df["fuel_city_hwy_ratio"] = (
    df["fuel_city"] / df["fuel_hwy"].replace(0, np.nan)
).fillna(1.4)

# label-encode categoricals
categorical_cols  = ["make", "vehicle_class", "transmission_type", "fuel_type"]
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[f"{col}_encoded"] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le


# ── 4. Features & target ────────────────────────────────────────────────────

FEATURES = [
    "engine_size", "cylinders",
    "fuel_city", "fuel_hwy", "fuel_city_hwy_ratio",
    "transmission_speeds",
    "make_encoded", "vehicle_class_encoded",
    "transmission_type_encoded", "fuel_type_encoded",
]

X = df[FEATURES].values
y = df["co2_emissions"].values

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)


# ── 5. Train & evaluate ─────────────────────────────────────────────────────

def mape(y_true, y_pred):
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

models = {
    "Ridge Regression":  Ridge(alpha=1.0),
    "Random Forest":     RandomForestRegressor(
                             n_estimators=200, max_depth=15,
                             random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(
                             n_estimators=200, max_depth=5,
                             learning_rate=0.1, random_state=42),
}

results = {}
print("\n" + "=" * 62)
print(f"{'Model':<25} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'MAPE%':>8}")
print("=" * 62)

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    results[name] = {
        "MAE":   mean_absolute_error(y_test, preds),
        "RMSE":  float(np.sqrt(mean_squared_error(y_test, preds))),
        "R2":    r2_score(y_test, preds),
        "MAPE":  mape(y_test, preds),
        "model": model,
        "preds": preds,
    }
    r = results[name]
    print(f"{name:<25} {r['MAE']:>8.2f} {r['RMSE']:>8.2f} {r['R2']:>8.4f} {r['MAPE']:>8.2f}%")

print("=" * 62)

best_name  = max(results, key=lambda k: results[k]["R2"])
best       = results[best_name]
print(f"\nBest model: {best_name}")
print(f"  MAE:  {best['MAE']:.2f} g/km")
print(f"  RMSE: {best['RMSE']:.2f} g/km")
print(f"  R²:   {best['R2']:.4f}")
print(f"  MAPE: {best['MAPE']:.2f}%")

# feature importance
best_model = best["model"]
if hasattr(best_model, "feature_importances_"):
    fi = dict(zip(FEATURES, best_model.feature_importances_))
elif hasattr(best_model, "coef_"):
    fi = dict(zip(FEATURES, np.abs(best_model.coef_)))
else:
    fi = {}


# ── 6. Save artifacts ───────────────────────────────────────────────────────

os.makedirs("models", exist_ok=True)

fuel_type_labels = {
    "X": "Regular Gasoline", "Z": "Premium Gasoline",
    "D": "Diesel", "E": "Ethanol E85", "N": "Natural Gas",
}

co2_series = df["co2_emissions"]
data_stats = {
    "makes":              sorted(df["make"].unique().tolist()),
    "vehicle_classes":    sorted(df["vehicle_class"].unique().tolist()),
    "transmission_types": sorted(df["transmission_type"].unique().tolist()),
    "fuel_types":         sorted(df["fuel_type"].unique().tolist()),
    "fuel_type_labels":   fuel_type_labels,
    "engine_size_range":  (float(df["engine_size"].min()), float(df["engine_size"].max())),
    "cylinders_options":  sorted(df["cylinders"].unique().tolist()),
    "fuel_city_range":    (float(df["fuel_city"].min()), float(df["fuel_city"].max())),
    "fuel_hwy_range":     (float(df["fuel_hwy"].min()), float(df["fuel_hwy"].max())),
    "co2_percentiles":    {p: float(np.percentile(co2_series, p)) for p in [10, 25, 50, 75, 90]},
    "co2_by_make":         df.groupby("make")["co2_emissions"].mean().round(1).to_dict(),
    "co2_by_vehicle_class":df.groupby("vehicle_class")["co2_emissions"].mean().round(1).to_dict(),
    "co2_by_fuel_type":    df.groupby("fuel_type")["co2_emissions"].mean().round(1).to_dict(),
    "all_co2":             co2_series.tolist(),
    "df_for_plots":        df[["make", "vehicle_class", "fuel_type",
                                "engine_size", "cylinders",
                                "fuel_city", "fuel_hwy",
                                "co2_emissions"]].to_dict("records"),
}

metrics_payload = {
    name: {k: v for k, v in m.items() if k not in ("model", "preds")}
    for name, m in results.items()
}
metrics_payload.update({
    "best_model_name":  best_name,
    "feature_importance": fi,
    "feature_names":    FEATURES,
    "y_test":           y_test,
    "y_pred":           best["preds"],
})

artifacts = {
    "models/co2_model.pkl":     best_model,
    "models/scaler.pkl":        scaler,
    "models/feature_names.pkl": FEATURES,
    "models/label_encoders.pkl":label_encoders,
    "models/model_metrics.pkl": metrics_payload,
    "models/data_stats.pkl":    data_stats,
}

for path, obj in artifacts.items():
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"  saved {path}  ({os.path.getsize(path)/1024:.1f} KB)")

print("\nAll done. Run:  streamlit run streamlit_app.py")
