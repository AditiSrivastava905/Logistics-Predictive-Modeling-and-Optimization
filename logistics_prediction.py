# ============================================================
# WEEK 4: PREDICTIVE MODELING AND OPTIMIZATION IN LOGISTICS
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ------------------------------------------------------------
# 1. PROJECT SETTINGS
# ------------------------------------------------------------

np.random.seed(42)

os.makedirs("visualizations", exist_ok=True)
os.makedirs("data", exist_ok=True)

print("=" * 60)
print("LOGISTICS PREDICTIVE MODELING AND OPTIMIZATION")
print("=" * 60)


# ------------------------------------------------------------
# 2. CREATE LOGISTICS DATASET
# ------------------------------------------------------------

n = 500

shipping_modes = np.random.choice(
    ["Road", "Rail", "Air", "Sea"],
    n,
    p=[0.55, 0.18, 0.17, 0.10]
)

regions = np.random.choice(
    ["North", "South", "East", "West"],
    n
)

shipment_volume = np.random.randint(20, 501, n)
distance = np.random.randint(30, 1501, n)
scheduled_days = np.random.randint(1, 9, n)

# Transportation cost rates for different modes
cost_rates = pd.Series(shipping_modes).map({
    "Road": 0.055,
    "Rail": 0.038,
    "Air": 0.160,
    "Sea": 0.025
}).values

transportation_cost = np.maximum(
    50,
    100
    + distance * shipment_volume * cost_rates / 100
    + np.random.normal(0, 70, n)
)

# Mode-related delivery-time effect
mode_effect = pd.Series(shipping_modes).map({
    "Road": 1.0,
    "Rail": 2.0,
    "Air": -0.8,
    "Sea": 3.0
}).values

# Simulated delivery time
delivery_time = np.maximum(
    0.5,
    1.2
    + distance / 500
    + mode_effect
    + shipment_volume / 800
    + np.random.normal(0, 1.0, n)
)

df = pd.DataFrame({
    "Region": regions,
    "Shipping_Mode": shipping_modes,
    "Shipment_Volume": shipment_volume,
    "Distance_km": distance,
    "Scheduled_Shipment_Days": scheduled_days,
    "Transportation_Cost": transportation_cost,
    "Delivery_Time_days": delivery_time
})

# Save dataset
df.to_csv("data/logistics_dataset.csv", index=False)

print("\nDataset created successfully!")
print("Number of records:", len(df))
print("\nFirst 5 records:")
print(df.head())


# ------------------------------------------------------------
# 3. BASIC DATA INFORMATION
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print("\nDataset shape:", df.shape)

print("\nMissing values:")
print(df.isnull().sum())

print("\nStatistical summary:")
print(df.describe())


# ------------------------------------------------------------
# 4. DEFINE FEATURES AND TARGET
# ------------------------------------------------------------

X = df.drop(columns=["Delivery_Time_days"])
y = df["Delivery_Time_days"]

categorical_features = [
    "Region",
    "Shipping_Mode"
]

numerical_features = [
    "Shipment_Volume",
    "Distance_km",
    "Scheduled_Shipment_Days",
    "Transportation_Cost"
]


# ------------------------------------------------------------
# 5. PREPROCESSING PIPELINE
# ------------------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            SimpleImputer(strategy="median"),
            numerical_features
        ),
        (
            "cat",
            Pipeline([
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent")
                ),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore")
                )
            ]),
            categorical_features
        )
    ]
)


# ------------------------------------------------------------
# 6. TRAIN-TEST SPLIT
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# ------------------------------------------------------------
# 7. LINEAR REGRESSION MODEL
# ------------------------------------------------------------

linear_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])

linear_model.fit(X_train, y_train)

linear_predictions = linear_model.predict(X_test)


# ------------------------------------------------------------
# 8. RANDOM FOREST MODEL
# ------------------------------------------------------------

random_forest_model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "model",
        RandomForestRegressor(
            n_estimators=150,
            max_depth=12,
            random_state=42,
            n_jobs=-1
        )
    )
])

random_forest_model.fit(X_train, y_train)

rf_predictions = random_forest_model.predict(X_test)


# ------------------------------------------------------------
# 9. MODEL EVALUATION FUNCTION
# ------------------------------------------------------------

def evaluate_model(name, actual, predicted):

    mae = mean_absolute_error(actual, predicted)

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    r2 = r2_score(actual, predicted)

    return {
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


linear_results = evaluate_model(
    "Linear Regression",
    y_test,
    linear_predictions
)

rf_results = evaluate_model(
    "Random Forest",
    y_test,
    rf_predictions
)

results = pd.DataFrame([
    linear_results,
    rf_results
])

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(results.round(3))


# ------------------------------------------------------------
# 10. CROSS-VALIDATION
# ------------------------------------------------------------

cv_scores = -cross_val_score(
    random_forest_model,
    X,
    y,
    cv=5,
    scoring="neg_mean_absolute_error",
    n_jobs=-1
)

print("\n" + "=" * 60)
print("5-FOLD CROSS-VALIDATION")
print("=" * 60)

print("Fold MAE scores:", cv_scores.round(3))
print("Mean CV MAE:", round(cv_scores.mean(), 3))


# ------------------------------------------------------------
# 11. VISUALIZATION 1
# ACTUAL VS PREDICTED - LINEAR REGRESSION
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    linear_predictions,
    alpha=0.6
)

minimum = min(
    y_test.min(),
    linear_predictions.min()
)

maximum = max(
    y_test.max(),
    linear_predictions.max()
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum]
)

plt.xlabel("Actual Delivery Time (days)")
plt.ylabel("Predicted Delivery Time (days)")
plt.title("Linear Regression: Actual vs Predicted")

plt.tight_layout()

plt.savefig(
    "visualizations/linear_actual_predicted.png",
    dpi=200
)

plt.show()


# ------------------------------------------------------------
# 12. VISUALIZATION 2
# ACTUAL VS PREDICTED - RANDOM FOREST
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    rf_predictions,
    alpha=0.6
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum]
)

plt.xlabel("Actual Delivery Time (days)")
plt.ylabel("Predicted Delivery Time (days)")
plt.title("Random Forest: Actual vs Predicted")

plt.tight_layout()

plt.savefig(
    "visualizations/random_forest_actual_predicted.png",
    dpi=200
)

plt.show()


# ------------------------------------------------------------
# 13. VISUALIZATION 3
# MODEL RMSE COMPARISON
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.bar(
    results["Model"],
    results["RMSE"]
)

plt.ylabel("RMSE (days)")
plt.xlabel("Model")
plt.title("Model RMSE Comparison")

plt.tight_layout()

plt.savefig(
    "visualizations/model_rmse_comparison.png",
    dpi=200
)

plt.show()


# ------------------------------------------------------------
# 14. OPTIMIZATION SCENARIO
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("LOGISTICS OPTIMIZATION SCENARIO")
print("=" * 60)

# Create 50 representative shipments
rng = np.random.default_rng(7)

scenario = pd.DataFrame({
    "Region": rng.choice(
        ["North", "South", "East", "West"],
        50
    ),

    "Shipment_Volume": rng.integers(
        50,
        451,
        50
    ),

    "Distance_km": rng.integers(
        100,
        1401,
        50
    ),

    "Scheduled_Shipment_Days": rng.integers(
        2,
        8,
        50
    )
})

mode_rates = {
    "Road": 0.055,
    "Rail": 0.038,
    "Air": 0.160,
    "Sea": 0.025
}

optimization_results = []

for mode in mode_rates:

    temp = scenario.copy()

    temp["Shipping_Mode"] = mode

    temp["Transportation_Cost"] = np.maximum(
        50,
        100
        + temp["Distance_km"]
        * temp["Shipment_Volume"]
        * mode_rates[mode]
        / 100
    )

    temp["Predicted_Delivery_Time"] = (
        random_forest_model.predict(
            temp[X.columns]
        )
    )

    temp["Mode"] = mode

    optimization_results.append(
        temp[
            [
                "Mode",
                "Predicted_Delivery_Time",
                "Transportation_Cost"
            ]
        ]
    )


optimization_df = pd.concat(
    optimization_results,
    ignore_index=True
)

mode_summary = (
    optimization_df
    .groupby("Mode")
    .agg(
        Average_Predicted_Delivery=(
            "Predicted_Delivery_Time",
            "mean"
        ),
        Average_Transportation_Cost=(
            "Transportation_Cost",
            "mean"
        )
    )
    .round(2)
)

print("\nTransportation Mode Comparison:")
print(mode_summary)


# ------------------------------------------------------------
# 15. OPTIMIZATION VISUALIZATION
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.bar(
    mode_summary.index,
    mode_summary["Average_Predicted_Delivery"]
)

plt.xlabel("Transportation Mode")
plt.ylabel("Predicted Delivery Time (days)")
plt.title("Predicted Delivery Time by Transportation Mode")

plt.tight_layout()

plt.savefig(
    "visualizations/optimization_delivery_time.png",
    dpi=200
)

plt.show()


# ------------------------------------------------------------
# 16. COST COMPARISON
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.bar(
    mode_summary.index,
    mode_summary["Average_Transportation_Cost"]
)

plt.xlabel("Transportation Mode")
plt.ylabel("Average Transportation Cost")
plt.title("Transportation Cost by Mode")

plt.tight_layout()

plt.savefig(
    "visualizations/optimization_cost.png",
    dpi=200
)

plt.show()


# ------------------------------------------------------------
# 17. SAVE MODEL RESULTS
# ------------------------------------------------------------

results.to_csv(
    "model_evaluation_results.csv",
    index=False
)

mode_summary.to_csv(
    "transportation_optimization_results.csv"
)


# ------------------------------------------------------------
# 18. FINAL PROJECT SUMMARY
# ------------------------------------------------------------

best_model = results.loc[
    results["RMSE"].idxmin()
]

print("\n" + "=" * 60)
print("FINAL PROJECT SUMMARY")
print("=" * 60)

print(
    "\nBest Model:",
    best_model["Model"]
)

print(
    "MAE:",
    round(best_model["MAE"], 3),
    "days"
)

print(
    "RMSE:",
    round(best_model["RMSE"], 3),
    "days"
)

print(
    "R2:",
    round(best_model["R2"], 3)
)

print(
    "\nMean Cross-Validation MAE:",
    round(cv_scores.mean(), 3),
    "days"
)

print("\nProject completed successfully!")

print("\nGenerated files:")
print("- data/logistics_dataset.csv")
print("- model_evaluation_results.csv")
print("- transportation_optimization_results.csv")
print("- visualizations/")