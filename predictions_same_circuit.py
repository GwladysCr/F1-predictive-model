predict = 2025
circuit = 'Silverstone'


import fastf1
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import json
import requests
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import xgboost as xgb


fastf1.Cache.enable_cache('__pycache__')

team_pts = {
    "McLaren": 800, "Mercedes": 459, "Red Bull": 426, "Williams": 137, "Ferrari": 382,
    "Haas": 73, "Aston Martin": 80, "Kick Sauber": 68, "Racing Bulls": 92, "Alpine": 22
}

driver_team = {
    "VER": "Red Bull", "NOR": "McLaren", "PIA": "McLaren", "LEC": "Ferrari", "RUS": "Mercedes",
    "HAM": "Ferrari", "GAS": "Alpine", "ALO": "Aston Martin", "TSU": "Racing Bulls",
    "SAI": "Williams", "HUL": "Kick Sauber", "OCO": "Alpine", "STR": "Aston Martin"
}


max_pts = max(team_pts.values())
team_score = {team: (pts / max_pts) for team, pts in team_pts.items()}


models = {
    'Ridge': Ridge(alpha=0.5),
    'Lasso': Ridge(alpha=2.0),
    'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_leaf=2, min_samples_split=5, random_state=42),
    'XGBoost': xgb.XGBRegressor(n_estimators=100, learning_rate=0.6, max_depth=5, reg_alpha = 5, reg_lambda = 10, random_state=42),
}


def get_weather(city, race_date):

    geo = requests.get("https://geocoding-api.open-meteo.com/v1/search", params={"name": city}, timeout=5).json()
    lat = geo["results"][0]["latitude"]
    lon = geo["results"][0]["longitude"]
    
    today = datetime.now().date()
    is_future = race_date > today
    
    if is_future:
        url = "https://api.open-meteo.com/v1/forecast"
    else:
        url = "https://archive-api.open-meteo.com/v1/archive"
        
    params = {
        'latitude': lat,
        'longitude': lon,
        'start_date': str(race_date),
        'end_date': str(race_date),
        'hourly': 'temperature_2m,wind_speed_10m,precipitation',
    }
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()
        
    hourly = data['hourly']
    race_hours = slice(13, 18)
    temp_values = hourly.get('temperature_2m', [])
    wind_values = hourly.get('wind_speed_10m', [])
    prec_values = hourly.get('precipitation', [])

    avg_temp = np.mean(temp_values[race_hours]) if np.mean(temp_values[race_hours]) != 0 else 20.0
    avg_wind_speed = np.mean(wind_values[race_hours]) if np.mean(wind_values[race_hours]) != 0 else 10.0
    avg_rain = np.mean(prec_values[race_hours])
                    
    return {'temperature': float(avg_temp), 'wind_speed': float(avg_wind_speed), 'rain': float(avg_rain)}


def calculate_clean_air_pace(driver_laps):
    timelaps = driver_laps['LapTime'].dt.total_seconds()
    threshold = timelaps.quantile(0.2)
    clean_laps = timelaps[timelaps <= threshold]
    if len(clean_laps) > 0:
        return clean_laps.mean()
    return timelaps.min()



def get_historical_data(circuit_name, year_to_predict):
    all_data = []
    years = [n for n in range(year_to_predict - 3, year_to_predict)]
         
    for year in years:
        schedule = fastf1.get_event_schedule(year)
             
        event = None
        for idx, row in schedule.iterrows():
            if circuit_name.lower() in row['Location'].lower() or circuit_name.lower() in row['EventName'].lower():
                event = row
                break
             
        if event is None:
            print(f"  Circuit non trouvé en {year}")
            continue
             
        # Charger qualification
        quali = fastf1.get_session(year, event['EventName'], 'Q')
        quali.load()
             
        # Charger course
        race = fastf1.get_session(year, event['EventName'], 'R')
        race.load()
             
        # Donnés météo
        race_date = event['EventDate'].date()
        weather_data = get_weather(event['Location'], race_date)

        for driver_abbr in race.laps['Driver'].unique():
            # Données de quali
            driver_quali_laps = quali.laps[quali.laps['Driver'] == driver_abbr]
            if len(driver_quali_laps) == 0:
                continue
                 
            best_quali = driver_quali_laps['LapTime'].min().total_seconds()
                 
            # Données de course
            driver_race_laps = race.laps[race.laps['Driver'] == driver_abbr]
                 
            # Clean Airpace
            avg_airpace = calculate_clean_air_pace(driver_quali_laps)
                 
            laps = driver_race_laps['LapTime'].dt.total_seconds()
            laps = laps.dropna()
            avg_laptime = np.mean(laps)
                 
            data_point = {
                'year': year,
                'driver': driver_abbr,
                'quali_time': best_quali,
                'team_points': team_score.get(driver_team.get(driver_abbr, ""), 0.5),
                'clean_airpace': avg_airpace,
                'temperature': weather_data['temperature'],
                'wind_speed': weather_data['wind_speed'],
                'rain': weather_data['rain'],
                'avg_laptime': avg_laptime
            }
                 
            all_data.append(data_point)
            
    df = pd.DataFrame(all_data)
    return df


def get_prediction_data(circuit_name, year_to_predict):
    
    schedule = fastf1.get_event_schedule(year_to_predict)
    
    event = None
    for idx, row in schedule.iterrows():
        if circuit_name.lower() in row['Location'].lower() or circuit_name.lower() in row['EventName'].lower():
            event = row
            break
    
    if event is None:
        raise ValueError(f"Circuit {circuit_name} non trouvé en {year_to_predict}")
    
    # Charger qualification
    quali = fastf1.get_session(year_to_predict, event['EventName'], 'Q')
    quali.load()
    
    # Météo
    race_date = event['EventDate'].date()
    weather_data = get_weather(event['Location'], race_date)
    
    quali_results = quali.results
    prediction_data = []
    
    for driver_abbr in quali.laps['Driver'].unique():
        driver_quali_laps = quali.laps[quali.laps['Driver'] == driver_abbr]
        
        if len(driver_quali_laps) == 0:
            continue
        
        best_quali = driver_quali_laps['LapTime'].min().total_seconds()
        avg_airpace = calculate_clean_air_pace(driver_quali_laps)
        
        data_point = {
            'driver': driver_abbr,
            'quali_time': best_quali,
            'clean_airpace': avg_airpace,
            'team_points': team_score.get(driver_team.get(driver_abbr, ""), 0.5),
            'temperature': weather_data['temperature'],
            'wind_speed': weather_data['wind_speed'],
            'rain': weather_data['rain'],
        }
        
        prediction_data.append(data_point)

    return pd.DataFrame(prediction_data)


def train_models(train_df):
    
    feature_cols = ['quali_time', 'clean_airpace', 'team_points', 
                     'temperature', 'wind_speed', 'rain',]
    train_df = train_df.dropna(subset=['avg_laptime'])
    X = train_df[feature_cols]
    y = train_df['avg_laptime']
    
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.1, random_state=39)
    
    trained_models = {}
    model_performance = {}
    feature_importance = {}
    
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[model_name] = model
        
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        mae_train = mean_absolute_error(y_train, y_pred_train)
        mae_test = mean_absolute_error(y_test, y_pred_test)
        r2_test = r2_score(y_test, y_pred_test)
        
        # Vérifier si le modèle extrapole correctement
        overfitting = abs(mae_train - mae_test) > 2.0
        overfitting_ratio = mae_test / mae_train if mae_train > 0 else 1.0
        
        print(f"\n{model_name:20s}")
        print(f"  Train MAE: {mae_train:.2f}s | Test MAE: {mae_test:.2f}s | R²: {r2_test:.3f}")
        
        if overfitting:
            print(f"  ⚠️  Overfitting détecté (différence MAE)")
        if overfitting_ratio > 2.5:
            print(f"  ⚠️  Overfitting détecté (ratio test/train = {overfitting_ratio:.2f})")
        if r2_test > 0.95 and len(y_train) < 100:
            print(f"  ⚠️  R² suspicieusement élevé sur petit dataset")
        
        
        model_performance[model_name] = {
            'MAE': mae_test,
            'R2': r2_test,
            'predictions': y_pred_test.tolist(),
            'actuals': y_test.tolist()
        }
        
        # Feature importance pour tree-based models
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_importance[model_name] = {
                'features': feature_cols,
                'importance': importances.tolist()
            }
            print(f"  Feature importance:")
            for feat, imp in zip(feature_cols, importances):
                print(f"    {feat}: {imp:.3f}")
        
    return [trained_models, model_performance, feature_importance]


def predict_race(predicted_df, trained_models, model_performance, feature_importance):
    
    feature_cols = ['quali_time', 'clean_airpace', 'team_points', 
                     'temperature', 'wind_speed', 'rain',]
    
    X = predicted_df[feature_cols]
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)
    
    
    all_predictions = {}
    for model_name, model in trained_models.items():
        all_predictions[model_name] = model.predict(X_imputed)
    
    prediction_rows = []
    for idx, (_, row) in enumerate(predicted_df.iterrows()):
        driver = row['driver']
        quali_time = row['quali_time']
        driver_pred = {'driver': driver, 'quali_time': quali_time}
        
        individual_preds = []
        
        for model_name in trained_models.keys():
            pred_time = all_predictions[model_name][idx]
            driver_pred[model_name] = pred_time
            
            # Pondérer selon la performance du modèle (R² score)
            model_r2 = model_performance[model_name]['R2']
            if model_r2 > 0:
                individual_preds.append((pred_time, model_r2))
        

        if individual_preds:
            total_weight = sum(weight for _, weight in individual_preds)
            if total_weight > 0:
                ensemble_pred = sum(pred * weight for pred, weight in individual_preds) / total_weight
                driver_pred['Ensemble'] = ensemble_pred
            else:
                driver_pred['Ensemble'] = np.mean([p for p, _ in individual_preds])
        
        prediction_rows.append(driver_pred)
    
    predictions_df = pd.DataFrame(prediction_rows)
    
    trained_models['Ensemble'] = 'weighted_average'

    podiums = {}
    available_models = ['Ridge', 'Lasso', 'RandomForest', 'XGBoost', 'Ensemble']
    
    for model_name in available_models:
        if model_name in predictions_df.columns:
            sorted_drivers = predictions_df.sort_values(by=model_name)
            podiums[model_name] = sorted_drivers.head(3)[['driver', model_name, 'quali_time']].to_dict('records')
    
    results = {
        'model_performance': model_performance,
        'feature_importance': feature_importance,
        'podiums': podiums,
        'all_predictions': predictions_df.to_dict('records'),
        'weather': {
            'rain': float(predicted_df['rain'].iloc[0]),
            'wind_speed': float(predicted_df['wind_speed'].iloc[0])
        }
    }
    
    return results


# MAIN

historical = get_historical_data(circuit, predict)


tot = train_models(historical)
predict_data = get_prediction_data(circuit, predict)

results = predict_race(predict_data, tot[0], tot[1], tot[2])

# Afficher les autres modèles
print("\n--- Modèles individuels ---")
for model_name, podium in results['podiums'].items():
    print(f"\n{model_name}:")
    for i, driver_data in enumerate(podium, 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
        quali = driver_data['quali_time']
        race = driver_data[model_name]
        diff_pct = ((race - quali) / quali) * 100
        print(f"  {medal} {i}. {driver_data['driver']:3s} - {race:.2f}s (Quali: {quali:.2f}s, +{diff_pct:.1f}%)")
