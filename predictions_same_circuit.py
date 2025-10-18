import fastf1
from fastf1 import events
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

from predictions_same_season import load_sessions

fastf1.Cache.enable_cache("f1_cache")

year = 2025
circuit = "Singapore"

team_pts = {
    "Red Bull": 290, "Mercedes": 325, "Ferrari": 298, "McLaren": 650,
    "Alpine": 20, "Aston Martin": 68, "Williams": 102, "Kick Sauber": 55,
    "Haas": 46, "Racing Bulls": 72
}

max_pts = max(team_pts.values())
team_score = {team: pts / max_pts for team, pts in team_pts.items()}

driver_team = {
    "VER": "Red Bull", "TSU": "Red Bull", "RUS": "Mercedes", "ANT": "Mercedes",
    "LEC": "Ferrari", "HAM": "Ferrari", "NOR": "McLaren", "PIA": "McLaren",
    "GAS": "Alpine", "COL": "Alpine", "ALO": "Aston Martin", "STR": "Aston Martin",
    "ALB": "Williams", "SAI": "Williams", "HUL": "Kick Sauber", "BOR": "Kick Sauber",
    "OCO": "Haas", "BEA": "Haas", "LAW": "Racing Bulls", "HAD": "Racing Bulls"
}

def get_round_number_by_name(year, partial_name):
    schedule = fastf1.get_event_schedule(year)
    for _, event in schedule.iterrows():
        if partial_name.lower() in event['EventName'].lower():
            return event['RoundNumber']

round1 = get_round_number_by_name(year - 1, circuit)
round2 = get_round_number_by_name(year - 2, circuit)

# Load race sessions
hist1 = fastf1.get_session(year - 1, round1, "R")
hist2 = fastf1.get_session(year - 2, round2, "R")
hist1.load()
hist2.load()

# Load qualifying sessions
quali1 = fastf1.get_session(year - 1, round1, "Q")
quali2 = fastf1.get_session(year - 2, round2, "Q")
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
def calculate_clean_air_pace(laps_df):
    clean_pace = {}
    for driver in laps_df["Driver"].unique():
        driver_laps = laps_df[laps_df["Driver"] == driver]["LapTime_s"]
        threshold = driver_laps.quantile(0.2)
        clean_laps = driver_laps[driver_laps <= threshold]
        if len(clean_laps) > 0:
            clean_pace[driver] = clean_laps.mean()
    return clean_pace

clean_pace1 = calculate_clean_air_pace(laps1)
clean_pace2 = calculate_clean_air_pace(laps2)

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

data1["TeamScore"] = data1["Driver"].map(lambda d: team_score.get(driver_team.get(d, ""), 0.5))
data2["TeamScore"] = data2["Driver"].map(lambda d: team_score.get(driver_team.get(d, ""), 0.5))

data_merged = pd.concat([data1, data2], ignore_index=True)
#data_merged = data1
data_merged.dropna(inplace=True)

# Prepare training and prediction datasets
feature_cols = ["QualiTime_s", "CleanAirPace_s", "TeamScore"]
X = data_merged[feature_cols]
y = data_merged["Total_s"]

imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost Model
xgb_model = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, objective="reg:squarederror")
xgb_model.fit(X_train, y_train)
xgb_scores = cross_val_score(xgb_model, X_train, y_train, cv=5, scoring="neg_mean_absolute_error")
print(f"Cross-validation MAE: {-xgb_scores.mean():.3f} ± {xgb_scores.std():.3f} seconds")

# Random Forest Model
rf_model = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_split=5, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring="neg_mean_absolute_error")
print(f"Cross-validation MAE: {-rf_scores.mean():.3f} ± {rf_scores.std():.3f} seconds")

# Evaluate on test
test_data = X_test.copy()
test_data["Driver"] = data_merged.loc[X_test.index, "Driver"]
test_data["XGBoost_Pred"] = xgb_model.predict(X_test)
test_data["RandomForest_Pred"] = rf_model.predict(X_test)
test_data["Ensemble_Pred"] = (test_data["XGBoost_Pred"] + test_data["RandomForest_Pred"]) / 2
test_data["Actual"] = y_test.values
print("________TEST RESULTS________")
print(test_data.sort_values("Ensemble_Pred")[[
    "Driver", "QualiTime_s", "CleanAirPace_s", "TeamScore", "XGBoost_Pred",
    "RandomForest_Pred", "Ensemble_Pred", "Actual"
]])
test_data['Residual_XGB'] = test_data['Actual'] - test_data['XGBoost_Pred']
test_data['Residual_RF'] = test_data['Actual'] - test_data['RandomForest_Pred']
test_data['Residual_Ens'] = test_data['Actual'] - test_data['Ensemble_Pred']
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Actual', y='Residual_XGB', data=test_data, label='XGBoost', color='green')
sns.scatterplot(x='Actual', y='Residual_RF', data=test_data, label='RandomForest', color='red')
sns.scatterplot(x='Actual', y='Residual_Ens', data=test_data, label='Ensemble', color='blue')
plt.axhline(0, linestyle='--', color='gray')
plt.title('Residual Plot')
plt.ylabel('Residual (Actual - Predicted)')
plt.xlabel('Actual Time (s)')
plt.legend()
plt.tight_layout()
plt.savefig('CIRCUIT_Residuals.png')


# Load quali of coming circuit
round = get_round_number_by_name(year, circuit)

_, quali = load_sessions(year, round)

quali_laps = quali.laps[["Driver", "LapTime"]].copy()
quali_laps.dropna(subset=["LapTime"], inplace=True)
quali_best = quali_laps.groupby("Driver")["LapTime"].min().dt.total_seconds()
for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
        if col in quali_laps.columns:
            quali_laps[f"{col}_s"] = quali_laps[col].dt.total_seconds()
clean_pace = calculate_clean_air_pace(quali_laps)

pred_df = pd.DataFrame({"Driver": quali_best.index})
pred_df["QualiTime_s"] = pred_df["Driver"].map(quali_best)
pred_df["CleanAirPace_s"] = pred_df["Driver"].map(clean_pace)
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
print("\nPredicted race pace order for", circuit)
print(pred_df[["Driver", "QualiTime_s", "CleanAirPace_s", "TeamScore", "Ensemble_Pred"]])


# -------------------PLOTTING------------------------------#
def plot_feature_importance(model, feature_names, model_name, save_path=None):
    importances = model.feature_importances_
    importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    importance_df = importance_df.sort_values(by="Importance", ascending=False)

    plt.figure(figsize=(8, 5))
    sns.barplot(data=importance_df, x="Importance", y="Feature", palette="viridis", legend=False)
    plt.title(f"{model_name} Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved {model_name} feature importance plot to {save_path}")
    plt.close()

def plot_top3_drivers(predictions_df, pred_col, model_name, save_path=None):
    top3 = predictions_df.head(3)
    plt.figure(figsize=(6, 4))
    sns.barplot(data=top3, x=pred_col, y="Driver", palette="coolwarm", legend=False)
    plt.title(f"Top 3 Drivers by {model_name} Predicted Race Pace")
    plt.xlabel("Predicted Lap Time (s)")
    plt.ylabel("Driver")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved top 3 drivers plot ({model_name}) to {save_path}")
    plt.close()

# After your models are trained and predictions made, call:

plot_feature_importance(xgb_model, feature_cols, "XGBoost", "CIRCUIT_xgb_feature_importance.png")
plot_feature_importance(rf_model, feature_cols, "Random Forest", "CIRCUIT_rf_feature_importance.png")

plot_top3_drivers(pred_df, "XGBoost_Pred", "XGBoost", "CIRCUIT_xgb_top3_drivers.png")
plot_top3_drivers(pred_df, "RandomForest_Pred", "Random Forest", "CIRCUIT_rf_top3_drivers.png")
plot_top3_drivers(pred_df, "Ensemble_Pred", "Ensemble", "CIRCUIT_ens_top3_drivers.png")
