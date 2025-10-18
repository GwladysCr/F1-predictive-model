import fastf1
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import main_functions as mf

fastf1.Cache.enable_cache("f1_cache")

# Load race sessions
hist1 = fastf1.get_session(mf.year, mf.circuit-1, "R")
hist2 = fastf1.get_session(mf.year, mf.circuit-2, "R")
hist1.load()
hist2.load()

# Load qualifying sessions
quali1 = fastf1.get_session(mf.year, mf.circuit-1, "Q")
quali2 = fastf1.get_session(mf.year, mf.circuit-2, "Q")
quali1.load()
quali2.load()

# LapTimes
laps1 = hist1.laps[["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]].copy()
laps1.dropna(subset=["LapTime"], inplace=True)
laps2 = hist2.laps[["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]].copy()
laps2.dropna(subset=["LapTime"], inplace=True)
quali1_laps = quali1.laps[["Driver", "LapTime"]].copy()
quali1_laps.dropna(subset=["LapTime"], inplace=True)
quali2_laps = quali2.laps[["Driver", "LapTime"]].copy()
quali2_laps.dropna(subset=["LapTime"], inplace=True)

# Convert to seconds
for df in [laps1, laps2]:
    for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
        if col in df.columns:
            df[f"{col}_s"] = df[col].dt.total_seconds()

# Get qualifying times for each driver (best lap)
quali1_times = quali1_laps.groupby("Driver")["LapTime"].min().dt.total_seconds()
quali2_times = quali2_laps.groupby("Driver")["LapTime"].min().dt.total_seconds()

# Clean Air Pace (20% best laps)
clean_pace1 = mf.calculate_clean_air_pace(laps1)
clean_pace2 = mf.calculate_clean_air_pace(laps2)

# Aggregate Sector Times
data1 = laps1.groupby("Driver").agg({
    "LapTime_s": "mean",
    "Sector1Time_s": "mean",
    "Sector2Time_s": "mean",
    "Sector3Time_s": "mean"
}).reset_index()

data2 = laps2.groupby("Driver").agg({
    "LapTime_s": "mean",
    "Sector1Time_s": "mean",
    "Sector2Time_s": "mean",
    "Sector3Time_s": "mean"
}).reset_index()

# Merge all data
data1["Total_s"] = data1["Sector1Time_s"] + data1["Sector2Time_s"] + data1["Sector3Time_s"]
data2["Total_s"] = data2["Sector1Time_s"] + data2["Sector2Time_s"] + data2["Sector3Time_s"]

data1["QualiTime_s"] = data1["Driver"].map(quali1_times)
data2["QualiTime_s"] = data2["Driver"].map(quali2_times)

data1["CleanAirPace_s"] = data1["Driver"].map(clean_pace1)
data2["CleanAirPace_s"] = data2["Driver"].map(clean_pace2)

data1["TeamScore"] = data1["Driver"].map(lambda d: mf.team_score.get(mf.driver_team.get(d, ""), 0.5))
data2["TeamScore"] = data2["Driver"].map(lambda d: mf.team_score.get(mf.driver_team.get(d, ""), 0.5))

data_merged = pd.concat([data1, data2], ignore_index=True)
#data_merged = data1
data_merged.dropna(inplace=True)

# Split train/test
feature_cols = ["QualiTime_s", "CleanAirPace_s", "TeamScore"]
X = data_merged[feature_cols]
y = data_merged["Total_s"]
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)


# Train models
xgb_model = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, objective="reg:squarederror")
xgb_model.fit(X_train, y_train)
rf_model = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_split=5, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# Cross‑validation on training
xgb_scores = cross_val_score(xgb_model, X_train, y_train, cv=5, scoring="neg_mean_absolute_error")
rf_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring="neg_mean_absolute_error")
print("XGBoost CV MAE:", -xgb_scores.mean(), "±", xgb_scores.std())
print("RF CV MAE:", -rf_scores.mean(), "±", rf_scores.std())


# Evaluate on test
test_data = pd.DataFrame({
    "Actual": y_test,
    "XGBoost_Pred": xgb_model.predict(X_test),
    "RandomForest_Pred": rf_model.predict(X_test),
    "Ensemble_Pred": (xgb_model.predict(X_test) + rf_model.predict(X_test)) / 2
})
mf.print_residuals(test_data, "SEASON_Residuals.png")



# Load quali of coming circuit

quali = fastf1.get_session(mf.year, mf.circuit, "Q")

quali.load()
quali_laps = quali.laps[["Driver", "LapTime"]].copy()
quali_laps.dropna(subset=["LapTime"], inplace=True)
quali_best = quali_laps.groupby("Driver")["LapTime"].min().dt.total_seconds()

for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
        if col in quali_laps.columns:
            quali_laps[f"{col}_s"] = quali_laps[col].dt.total_seconds()
clean_pace = mf.calculate_clean_air_pace(quali_laps)

pred_df = pd.DataFrame({"Driver": quali_best.index})
pred_df["QualiTime_s"] = pred_df["Driver"].map(quali_best)
pred_df["CleanAirPace_s"] = pred_df["Driver"].map(clean_pace)
pred_df["TeamScore"] = pred_df["Driver"].map(lambda d: mf.team_score.get(mf.driver_team.get(d, ""), .5))

X_pred = pred_df[feature_cols]
X_pred_im = imputer.transform(X_pred)

pred_df["XGBoost_Pred"] = xgb_model.predict(X_pred_im)
pred_df["RandomForest_Pred"] = rf_model.predict(X_pred_im)
pred_df["Ensemble_Pred"] = (pred_df["XGBoost_Pred"] + pred_df["RandomForest_Pred"]) / 2

# Sort by predicted pace (lower is faster)
pred_df = pred_df.sort_values("Ensemble_Pred")
print("\nPredicted race pace order for", mf.circuit)
print(pred_df[["Driver", "QualiTime_s", "CleanAirPace_s", "TeamScore", "Ensemble_Pred"]])


# -------------------PLOTTING------------------------------#
mf.plot_feature_importance(xgb_model, feature_cols, "XGBoost", "SEASON_xgb_feature_importance.png")
mf.plot_feature_importance(rf_model, feature_cols, "Random Forest", "SEASON_rf_feature_importance.png")

mf.plot_top3_drivers(pred_df, "XGBoost_Pred", "XGBoost", "SEASON_xgb_top3_drivers.png")
mf.plot_top3_drivers(pred_df, "RandomForest_Pred", "Random Forest", "SEASON_rf_top3_drivers.png")
mf.plot_top3_drivers(pred_df, "Ensemble_Pred", "Ensemble", "SEASON_ens_top3_drivers.png")
