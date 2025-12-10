# F1 Race Lap Time Predictor - 2025

This project predicts Formula 1 race lap times for upcoming Grand Prix events using historical data, qualifying results, weather conditions, and team/driver statistics. It leverages machine learning models including **Random Forest**, **Ridge Regression**, and **XGBoost**.

---

## **Features**

- Pulls historical F1 race data using [FastF1](https://theoehrly.github.io/Fast-F1/).  

- Collects weather data (temperature, wind, precipitation) for race days using the [Open-Meteo API](https://open-meteo.com/).  

- Computes a “clean air pace” metric from qualifying laps to estimate optimal lap times.  

- Supports multiple machine learning models:
  - Ridge Regression
  - Random Forest Regressor
  - XGBoost Regressor
  - Ensemble of all models weighted by R² performance

- Outputs **podium predictions** with predicted lap times and percentage differences from qualifying.

---

## **Installation**

1. Clone the repository:

```bash
git clone https://github.com/your-username/f1-lap-predictor.git
cd f1-lap-predictor
````

2. Install required Python packages:

```bash
pip install fastf1 pandas numpy scikit-learn xgboost requests
```

---

## **Usage**

1. Set the circuit and year to predict:

```python
predict = 2025
circuit = 'Monaco'
```

2. Run the main script:

```bash
python predict_f1.py
```

3. The script will:

* Load historical data for the last 3 seasons for the selected circuit.
* Train machine learning models on historical lap times.
* Fetch qualifying and weather data for the target race.
* Predict lap times for each driver.
* Display podium predictions for each model and the ensemble.

Example output:

```
--- Modèles individuels ---

RandomForest:
  🥇 1. NOR - 90.31s (Quali: 82.41s, +9.6%)
  🥈 2. PIA - 90.31s (Quali: 82.44s, +9.6%)
  🥉 3. RUS - 90.31s (Quali: 82.64s, +9.3%)
```

---

## **Project Structure**

* `predict_f1.py` : Main script with data loading, model training, and prediction functions.
* `__pycache__/` : Cache directory for FastF1 session data.

---

## **Configuration**

* `team_pts` : Points per team used to scale team performance in predictions.
* `driver_team` : Mapping of drivers to their teams.
* `models` : Dictionary of ML models with hyperparameters. Can be customized.

---

## **Notes**

* Weather data is retrieved automatically from Open-Meteo. For future races, forecasts are used; for past races, historical weather data is pulled.
* The ensemble prediction is a **weighted average** of all models based on their R² score.
* Ensure internet connectivity for FastF1 and Open-Meteo API calls.

---

## **License**

This project is released under the MIT License.

---

## **Acknowledgements**

* [FastF1](https://github.com/theoehrly/Fast-F1) for providing Formula 1 telemetry and session data.
* [Open-Meteo](https://open-meteo.com/) for free weather data API.


