import fastf1
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

fastf1.Cache.enable_cache("f1_cache")

season = 2025
target_circuit = 18

team_pts = {"Red Bull": 290, "Mercedes": 325, "Ferrari": 298, "McLaren": 650, "Alpine": 20, "Aston Martin": 68, "Williams": 102, 
            "Kick Sauber": 55, "Haas": 46, "Racing Bulls": 72} 
max_pts = max(team_pts.values()) 
team_score = {team: pts / max_pts for team, pts in team_pts.items()} 

driver_team = {"VER": "Red Bull", "TSU": "Red Bull", "RUS": "Mercedes", "ANT": "Mercedes", "LEC": "Ferrari", "HAM": "Ferrari",
                "NOR": "McLaren", "PIA": "McLaren", "GAS": "Alpine", "COL": "Alpine", "ALO": "Aston Martin", "STR": "Aston Martin", 
                "ALB": "Williams", "SAI": "Williams", "HUL": "Kick Sauber", "BOR": "Kick Sauber", "OCO": "Haas", "BEA": "Haas", 
                "LAW": "Racing Bulls", "HAD": "Racing Bulls"}


def load_sessions(year, circuit):
    race = fastf1.get_session(year, circuit, "R")
    quali = fastf1.get_session(year, circuit, "Q")
    race.load()
    quali.load()
    return race, quali

def build_features_from_session(race_sess, quali_sess, team_score_map, driver_team_map):
    # Extract lap data
    laps = race_sess.laps[["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]].copy()
    laps.dropna(subset=["LapTime"], inplace=True)
    for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
        if col in laps.columns:
            laps[f"{col}_s"] = laps[col].dt.total_seconds()

    # Quali best time
    quali_laps = quali_sess.laps[["Driver", "LapTime"]].copy()
    quali_laps.dropna(subset=["LapTime"], inplace=True)
    quali_best = quali_laps.groupby("Driver")["LapTime"].min().dt.total_seconds()

    # Clean air pace (top 20%) for race laps
    def calculate_clean_air_pace(laps_df):
        clean = {}
        for d in laps_df["Driver"].unique():
            arr = laps_df[laps_df["Driver"] == d]["LapTime_s"]
            if len(arr) == 0:
                continue
            thr = arr.quantile(0.2)
            top = arr[arr <= thr]
            if len(top) > 0:
                clean[d] = top.mean()
        return clean

    clean_pace = calculate_clean_air_pace(laps)

    # Aggregate per-driver features
    agg = laps.groupby("Driver").agg({
        "LapTime_s":"median",
        "Sector1Time_s":"mean",
        "Sector2Time_s":"mean",
        "Sector3Time_s":"mean"
    }).reset_index()
    agg["Total_s"] = agg["Sector1Time_s"] + agg["Sector2Time_s"] + agg["Sector3Time_s"]
    agg["QualiTime_s"] = agg["Driver"].map(quali_best)
    agg["CleanAirPace_s"] = agg["Driver"].map(clean_pace)
    agg["TeamScore"] = agg["Driver"].map(lambda d: team_score_map.get(driver_team_map.get(d, ""), 0.5))

    # Drop drivers missing any essential feature or target
    agg = agg.dropna()
    return agg


# -------------------MAIN------------------------------#
# Build full dataset for all prior races of the season
all_data = []
for circuit in range(target_circuit-3,target_circuit):
    race_sess, quali_sess = load_sessions(season, circuit)
    df_feat = build_features_from_session(race_sess, quali_sess, team_score, driver_team)
    df_feat["Circuit"] = circuit
    all_data.append(df_feat)

data_all = pd.concat(all_data, ignore_index=True)


# Split last race as test
test_circuit = target_circuit-1
train_data = data_all[data_all["Circuit"] != test_circuit].copy()
test_data = data_all[data_all["Circuit"] == test_circuit].copy()
latest_clean = test_data.set_index("Driver")["CleanAirPace_s"]

feature_cols = ["QualiTime_s", "CleanAirPace_s", "TeamScore"]
X_train = train_data[feature_cols]
y_train = train_data["Total_s"]
X_test = test_data[feature_cols]
y_test = test_data["Total_s"]

imputer = SimpleImputer(strategy="median")
X_train_im = imputer.fit_transform(X_train)
X_test_im = imputer.transform(X_test)


# Train models
xgb_model = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, objective="reg:squarederror")
xgb_model.fit(X_train_im, y_train)
rf_model = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_split=5, random_state=42, n_jobs=-1)
rf_model.fit(X_train_im, y_train)

# Cross‑validation on training
xgb_scores = cross_val_score(xgb_model, X_train_im, y_train, cv=5, scoring="neg_mean_absolute_error")
rf_scores = cross_val_score(rf_model, X_train_im, y_train, cv=5, scoring="neg_mean_absolute_error")
print("XGBoost CV MAE:", -xgb_scores.mean(), "±", xgb_scores.std())
print("RF CV MAE:", -rf_scores.mean(), "±", rf_scores.std())


# Evaluate on test
test_data = test_data.copy()
test_data["XGBoost_Pred"] = xgb_model.predict(X_test_im)
test_data["RandomForest_Pred"] = rf_model.predict(X_test_im)
test_data["Ensemble_Pred"] = (test_data["XGBoost_Pred"] + test_data["RandomForest_Pred"]) / 2
test_data["Actual"] = y_test.values
print("________TEST RESULTS________")
print(test_data.sort_values("Ensemble_Pred")[[
    "Driver", "QualiTime_s", "CleanAirPace_s", "TeamScore", "XGBoost_Pred",
    "RandomForest_Pred", "Ensemble_Pred", "Actual"
]])


# Load quali of coming circuit
_, us_quali = load_sessions(season, target_circuit)

quali_laps_us = us_quali.laps[["Driver", "LapTime"]].copy()
quali_laps_us.dropna(subset=["LapTime"], inplace=True)
quali_best_us = quali_laps_us.groupby("Driver")["LapTime"].min().dt.total_seconds()

pred_df = pd.DataFrame({"Driver": quali_best_us.index})
pred_df["QualiTime_s"] = pred_df["Driver"].map(quali_best_us)
pred_df["CleanAirPace_s"] = pred_df["Driver"].map(latest_clean)
pred_df["TeamScore"] = pred_df["Driver"].map(lambda d: team_pts.get(driver_team.get(d, ""), 0) / max_pts)

# Drop rows with missing features
pred_df = pred_df.dropna()

X_pred = pred_df[feature_cols]
X_pred_im = imputer.transform(X_pred)

pred_df["XGBoost_Pred"] = xgb_model.predict(X_pred_im)
pred_df["RandomForest_Pred"] = rf_model.predict(X_pred_im)
pred_df["Ensemble_Pred"] = (pred_df["XGBoost_Pred"] + pred_df["RandomForest_Pred"]) / 2

# Sort by predicted pace (lower is faster)
pred_df = pred_df.sort_values("Ensemble_Pred")
print("\nPredicted race pace order for", target_circuit)
print(pred_df[["Driver", "QualiTime_s", "CleanAirPace_s", "TeamScore", "Ensemble_Pred"]])


# -------------------PLOTTING------------------------------#
def plot_feature_importance(model, feature_names, model_name, save_path=None):
    importances = model.feature_importances_
    importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    importance_df = importance_df.sort_values(by="Importance", ascending=False)

    plt.figure(figsize=(8, 5))
    sns.barplot(data=importance_df, x="Importance", y="Feature", palette="viridis")
    plt.title(f"{model_name} Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved {model_name} feature importance plot to {save_path}")
    plt.close()

def plot_top3_drivers(predictions_df, pred_col, model_name, save_path=None):
    top3 = predictions_df.nsmallest(3, pred_col)
    plt.figure(figsize=(6, 4))
    sns.barplot(data=top3, x=pred_col, y="Driver", palette="coolwarm")
    plt.title(f"Top 3 Drivers by {model_name} Predicted Race Pace")
    plt.xlabel("Predicted Lap Time (s)")
    plt.ylabel("Driver")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved top 3 drivers plot ({model_name}) to {save_path}")
    plt.close()

plot_feature_importance(xgb_model, feature_cols, "XGBoost", "xgb_feature_importance.png")
plot_feature_importance(rf_model, feature_cols, "Random Forest", "rf_feature_importance.png")

plot_top3_drivers(pred_df, "XGBoost_Pred", "XGBoost", "xgb_top3_drivers.png")
plot_top3_drivers(pred_df, "RandomForest_Pred", "Random Forest", "rf_top3_drivers.png")
