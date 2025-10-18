import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


year = 2025
event = "Singapore"
circuit = 18

team_pts = {
    "Red Bull": 290, "Mercedes": 325, "Ferrari": 298, "McLaren": 650,
    "Alpine": 20, "Aston Martin": 68, "Williams": 102, "Kick Sauber": 55,
    "Haas": 46, "Racing Bulls": 72
}

max_pts = max(team_pts.values())
team_score = {team: (pts / max_pts) for team, pts in team_pts.items()}

driver_team = {
    "VER": "Red Bull", "TSU": "Red Bull", "RUS": "Mercedes", "ANT": "Mercedes",
    "LEC": "Ferrari", "HAM": "Ferrari", "NOR": "McLaren", "PIA": "McLaren",
    "GAS": "Alpine", "COL": "Alpine", "ALO": "Aston Martin", "STR": "Aston Martin",
    "ALB": "Williams", "SAI": "Williams", "HUL": "Kick Sauber", "BOR": "Kick Sauber",
    "OCO": "Haas", "BEA": "Haas", "LAW": "Racing Bulls", "HAD": "Racing Bulls"
}

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

# Print Residuals
def print_residuals(test_data, save_path=None):
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
    plt.savefig(save_path)


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