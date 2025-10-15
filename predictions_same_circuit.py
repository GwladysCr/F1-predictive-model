import fastf1
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

fastf1.Cache.enable_cache("f1_cache")

circuit = "Singapour Grand Prix"
year = 2025

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

# Load race sessions
hist1 = fastf1.get_session(year - 1, circuit, "R")
hist2 = fastf1.get_session(year - 2, circuit, "R")
hist1.load()
hist2.load()

# Load qualifying sessions
quali1 = fastf1.get_session(year - 1, circuit, "Q")
quali2 = fastf1.get_session(year - 2, circuit, "Q")
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
data_merged.dropna(inplace=True)

# Prepare training and prediction datasets
feature_cols = ["QualiTime_s", "CleanAirPace_s", "TeamScore"]
X = data_merged[feature_cols]
y = data_merged["Total_s"]

imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost Model
xgb_model = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    objective="reg:squarederror"
)
xgb_model.fit(X_train, y_train)
xgb_scores = cross_val_score(xgb_model, X_train, y_train, cv=5, scoring="neg_mean_absolute_error")
print(f"Cross-validation MAE: {-xgb_scores.mean():.3f} ± {xgb_scores.std():.3f} seconds")

# Random Forest Model
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
rf_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring="neg_mean_absolute_error")
print(f"Cross-validation MAE: {-rf_scores.mean():.3f} ± {rf_scores.std():.3f} seconds")

test_data = data_merged.loc[X_test.index].copy()
test_data["XGBoost_Pred"] = xgb_model.predict(X_test)
test_data["RandomForest_Pred"] = rf_model.predict(X_test)
test_data["Ensemble_Pred"] = (test_data["XGBoost_Pred"] + test_data["RandomForest_Pred"]) / 2
test_data["Actual"] = y_test

test_data.sort_values("XGBoost_Pred", inplace=True)
print(test_data)


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

# After your models are trained and predictions made, call:

plot_feature_importance(xgb_model, feature_cols, "XGBoost", "xgb_feature_importance_2.png")
plot_feature_importance(rf_model, feature_cols, "Random Forest", "rf_feature_importance_2.png")

plot_top3_drivers(test_data, "XGBoost_Pred", "XGBoost", "xgb_top3_drivers_2.png")
plot_top3_drivers(test_data, "RandomForest_Pred", "Random Forest", "rf_top3_drivers_2.png")
