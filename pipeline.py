
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

# Ensure dataset exists
from generate_chennai_data import generate_chennai_housing_dataset
df_raw = generate_chennai_housing_dataset()

# ==============================================================================
# STEP 1, 2 & 3: DATA INSPECTION, MISSING VALUES, OUTLIERS & LEAKAGE AUDIT
# ==============================================================================
print("\n--- STEP 3: MISSING VALUES & LEAKAGE AUDIT ---")
print("Missing Values per Column:\n", df_raw.isnull().sum()[df_raw.isnull().sum() > 0])

# Identify Leakage Features (Correlations > 0.90 or Post-Transaction Knowledge)
LEAKAGE_COLS = ["PRT_ID", "PRICE_PER_SQFT", "REG_FEE_LAKHS", "COMMIS_LAKHS"]
print(f"\nDropping Identifier & Leakage Columns: {LEAKAGE_COLS}")
df_clean = df_raw.drop(columns=LEAKAGE_COLS).copy()

# Cap extreme luxury outliers above 99th percentile to prevent MSE distortion
upper_cap = df_clean["PRICE_LAKHS"].quantile(0.99)
print(f"99th Percentile Price Cap applied at: ₹{upper_cap:.2f} Lakhs")
df_clean = df_clean[df_clean["PRICE_LAKHS"] <= upper_cap].reset_index(drop=True)

# ==============================================================================
# STEP 5: EDA VISUALISATIONS (Area, Location, Rooms, Property Age)
# ==============================================================================
sns.set_theme(style="whitegrid", palette="muted")
fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle("PropPulse Chennai — Exploratory Data Analysis (Step 5)", fontsize=16, fontweight="bold")

# 1. Area vs Price by BuildType
sns.scatterplot(
    data=df_clean.sample(600, random_state=42),
    x="INT_SQFT", y="PRICE_LAKHS", hue="BUILDTYPE", alpha=0.75, ax=axes[0, 0]
)
axes[0, 0].set_title("1. Interior Area (SqFt) vs Price (₹ Lakhs)")
axes[0, 0].set_xlabel("Interior Area (SqFt)")
axes[0, 0].set_ylabel("Price (₹ Lakhs)")

# 2. Location vs Price Boxplot
order_loc = df_clean.groupby("AREA")["PRICE_LAKHS"].median().sort_values(ascending=False).index
sns.boxplot(data=df_clean, x="AREA", y="PRICE_LAKHS", order=order_loc, ax=axes[0, 1], palette="Blues_r")
axes[0, 1].set_title("2. Price Distribution Across Chennai Corridors")
axes[0, 1].tick_params(axis="x", rotation=30)
axes[0, 1].set_xlabel("Locality")
axes[0, 1].set_ylabel("Price (₹ Lakhs)")

# 3. Bedrooms vs Price Violin Plot
sns.violinplot(data=df_clean, x="N_BEDROOM", y="PRICE_LAKHS", ax=axes[1, 0], palette="crest")
axes[1, 0].set_title("3. Price Distribution by Bedroom Count (BHK)")
axes[1, 0].set_xlabel("Number of Bedrooms (BHK)")
axes[1, 0].set_ylabel("Price (₹ Lakhs)")

# 4. Property Age vs Price Depreciation Trend
sns.regplot(
    data=df_clean.sample(600, random_state=42),
    x="PROPERTY_AGE", y="PRICE_LAKHS",
    scatter_kws={"alpha": 0.3, "color": "#475569"},
    line_kws={"color": "#dc2626", "linewidth": 2},
    ax=axes[1, 1]
)
axes[1, 1].set_title("4. Property Age Depreciation Curve")
axes[1, 1].set_xlabel("Property Age at Sale (Years)")
axes[1, 1].set_ylabel("Price (₹ Lakhs)")

plt.tight_layout()
plt.savefig("eda_visualisations.png", dpi=200)
print("[SAVED] 'eda_visualisations.png' (4-panel EDA chart)")

# ==============================================================================
# STEP 4: REPRODUCIBLE PREPROCESSING PIPELINE
# ==============================================================================
NUM_FEATURES = ["INT_SQFT", "N_BEDROOM", "N_BATHROOM", "N_ROOM", "PROPERTY_AGE", "DIST_MAINROAD"]
CAT_FEATURES = ["AREA", "BUILDTYPE", "PARK_FACIL", "SALE_COND"]
TARGET = "PRICE_LAKHS"

X = df_clean[NUM_FEATURES + CAT_FEATURES]
y = df_clean[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, NUM_FEATURES),
    ("cat", categorical_transformer, CAT_FEATURES)
])

# ==============================================================================
# STEP 6 & 7: BASELINE & REGRESSION MODEL COMPARISON (MAE & RMSE IN ₹ LAKHS)
# ==============================================================================
models = {
    "0. Baseline (Median Dummy)": DummyRegressor(strategy="median"),
    "1. Ridge Linear Regression": Ridge(alpha=1.0),
    "2. Random Forest Regressor": RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1),
    "3. Gradient Boosting (Champion)": GradientBoostingRegressor(
        loss="squared_error", n_estimators=250, learning_rate=0.06, max_depth=4, random_state=42
    )
}

results = []
trained_pipelines = {}

for name, model in models.items():
    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    
    results.append({
        "Model": name,
        "MAE (₹ Lakhs)": round(mae, 2),
        "RMSE (₹ Lakhs)": round(rmse, 2),
        "R² Score": round(r2, 4)
    })
    trained_pipelines[name] = pipe

results_df = pd.DataFrame(results)
print("\n--- STEP 6 & 7: MODEL BENCHMARK TABLE ---")
print(results_df.to_string(index=False))

# ==============================================================================
# STEP 9 PREP: QUANTILE REGRESSION FOR 90% UNCERTAINTY PREDICTION INTERVALS
# ==============================================================================
champion_pipe = trained_pipelines["3. Gradient Boosting (Champion)"]

lower_pipe = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", GradientBoostingRegressor(
        loss="quantile", alpha=0.05, n_estimators=250, learning_rate=0.06, max_depth=4, random_state=42
    ))
])
upper_pipe = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", GradientBoostingRegressor(
        loss="quantile", alpha=0.95, n_estimators=250, learning_rate=0.06, max_depth=4, random_state=42
    ))
])

lower_pipe.fit(X_train, y_train)
upper_pipe.fit(X_train, y_train)

# Check empirical coverage on test set
y_pred_point = champion_pipe.predict(X_test)
y_pred_low = lower_pipe.predict(X_test)
y_pred_high = upper_pipe.predict(X_test)
coverage = np.mean((y_test >= y_pred_low) & (y_test <= y_pred_high)) * 100
print(f"\n[UNCERTAINTY CHECK] 90% Prediction Interval Empirical Test Coverage: {coverage:.1f}%")

# ==============================================================================
# STEP 8: SUBGROUP ERROR ANALYSIS ACROSS PROPERTY TIERS & LOCALITIES
# ==============================================================================
eval_df = X_test.copy()
eval_df["Actual_Lakhs"] = y_test.values
eval_df["Predicted_Lakhs"] = y_pred_point
eval_df["Abs_Error_Lakhs"] = np.abs(eval_df["Actual_Lakhs"] - eval_df["Predicted_Lakhs"])
eval_df["APE_Percent"] = (eval_df["Abs_Error_Lakhs"] / eval_df["Actual_Lakhs"]) * 100

eval_df["Price_Tier"] = pd.cut(
    eval_df["Actual_Lakhs"],
    bins=[0, 65, 130, 1000],
    labels=["Budget (< ₹65L)", "Mid-Segment (₹65L - ₹1.3Cr)", "Luxury (> ₹1.3Cr)"]
)

tier_summary = eval_df.groupby("Price_Tier", observed=False).agg(
    Count=("Actual_Lakhs", "count"),
    MAE_Lakhs=("Abs_Error_Lakhs", "mean"),
    RMSE_Lakhs=("Abs_Error_Lakhs", lambda x: np.sqrt(np.mean(x**2))),
    MAPE_Percent=("APE_Percent", "mean")
).round(2)

print("\n--- STEP 8: SUBGROUP ERROR ANALYSIS BY PROPERTY TIER ---")
print(tier_summary)

# ==============================================================================
# STEP 10: PERMUTATION FEATURE IMPORTANCE & DIAGNOSTIC PLOTS
# ==============================================================================
perm_imp = permutation_importance(champion_pipe, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
imp_df = pd.DataFrame({
    "Feature": X_test.columns,
    "Importance_Mean": perm_imp.importances_mean,
    "Importance_Std": perm_imp.importances_std
}).sort_values("Importance_Mean", ascending=True)

fig2, axes2 = plt.subplots(1, 2, figsize=(15, 6))
fig2.suptitle("Model Diagnostics & Explainability (Steps 8 & 10)", fontsize=15, fontweight="bold")

# Plot Actual vs Predicted
axes2[0].scatter(eval_df["Actual_Lakhs"], eval_df["Predicted_Lakhs"], alpha=0.5, color="#2563eb", label="Predictions")
max_val = max(eval_df["Actual_Lakhs"].max(), eval_df["Predicted_Lakhs"].max())
axes2[0].plot([15, max_val], [15, max_val], "r--", lw=2, label="Ideal 1:1 Line")
axes2[0].set_title("Actual vs Predicted House Prices (₹ Lakhs)")
axes2[0].set_xlabel("Actual Price (₹ Lakhs)")
axes2[0].set_ylabel("Predicted Price (₹ Lakhs)")
axes2[0].legend()

# Plot Permutation Importance
axes2[1].barh(imp_df["Feature"], imp_df["Importance_Mean"], xerr=imp_df["Importance_Std"], color="#0d9488")
axes2[1].set_title("Permutation Importance (R² Drop on Shuffle)")
axes2[1].set_xlabel("Decrease in R² Score")

plt.tight_layout()
plt.savefig("model_diagnostics_and_xai.png", dpi=200)
print("[SAVED] 'model_diagnostics_and_xai.png'")

# Save Bundle for Streamlit Prototype
bundle = {
    "point_model": champion_pipe,
    "lower_model": lower_pipe,
    "upper_model": upper_pipe,
    "metrics": results_df.to_dict(orient="records"),
    "feature_importance": imp_df.sort_values("Importance_Mean", ascending=False).to_dict(orient="records"),
    "localities": sorted(df_clean["AREA"].unique().tolist()),
    "buildtypes": sorted(df_clean["BUILDTYPE"].unique().tolist()),
}
joblib.dump(bundle, "chennai_price_bundle.pkl")
print("[SAVED] 'chennai_price_bundle.pkl' ready for Streamlit app!")
