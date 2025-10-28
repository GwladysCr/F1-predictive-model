# F1 Race Pace Prediction
This project uses historical and current Formula 1 data to predict race pace for upcoming Grand Prix events. It leverages the FastF1 library to extract session data and applies machine learning models to forecast driver performance.  
Two complementary approaches are implemented:  
&nbsp; &nbsp;&nbsp; - Circuit-based analysis: Uses past editions of the same Grand Prix.  
&nbsp; &nbsp;&nbsp; - Season-based analysis: Uses recent races from the current season.  

# Project Structure
f1_race_pace_prediction/
│
├── prediction_same_circuit.py         # Prédiction basée sur l'historique du circuit
├── prediction_same_season.py          # Prédiction basée sur les courses précédentes de la saison
├── main_functions.py          # Fonctions utilitaires (visualisation, calculs, mapping)
├── f1_cache/                  # Cache FastF1 pour accélérer les chargements
├── README.md                  # Ce fichier


# Goal
Predict the average clean air race pace (lap time in seconds) for each driver in an upcoming race using:  
&nbsp; &nbsp;&nbsp; - Qualifying performance (QualiTime_s)  
&nbsp; &nbsp;&nbsp; - Clean air pace from previous races (CleanAirPace_s)  
&nbsp; &nbsp;&nbsp; - Team performance score (TeamScore)  

# Models Used
&nbsp; &nbsp;&nbsp; - XGBoost Regressor  
&nbsp; &nbsp;&nbsp; - Random Forest Regressor  
&nbsp; &nbsp;&nbsp; - Ensemble Model: average of both predictions  

Model performance is evaluated using:  
&nbsp; &nbsp;&nbsp; - Cross-validation (MAE)  
&nbsp; &nbsp;&nbsp; - Residual analysis  
&nbsp; &nbsp;&nbsp; - Feature importance plots  

# Visual Outputs
Each script generates:  
&nbsp; &nbsp;&nbsp; - Residual plot (Residuals.png)  
&nbsp; &nbsp;&nbsp; - Feature importance plots for each model  
&nbsp; &nbsp;&nbsp; - Top 3 predicted drivers by model  

# How to Use
  1. Run a prediction:  
&nbsp; &nbsp;&nbsp; - For circuit-based analysis: python predictions_same_circuit.py
&nbsp; &nbsp;&nbsp; - For season-based analysis: python predictions_same_season.py
	2. View results:  
&nbsp; &nbsp;&nbsp; - Predictions are printed in the console.  
&nbsp; &nbsp;&nbsp; - Plots are saved automatically.  

# Utility Functions
Located in main_functions.py:
&nbsp; &nbsp;&nbsp; - calculate_clean_air_pace: computes average of the fastest 20% laps  
&nbsp; &nbsp;&nbsp; - print_residuals: visualizes prediction errors  
&nbsp; &nbsp;&nbsp; - plot_feature_importance: shows model feature weights  
&nbsp; &nbsp;&nbsp; - plot_top3_drivers: highlights top predicted performers    

# Example Output
Predicted race pace order for Singapore
Driver   | QualiTimes | CleanAirPaces | TeamScore | Ensemble_Pred
-------- |-------------|----------------|-----------|----------------
NOR      | 89.123      | 90.456         | 1.000     | 90.321
VER      | 89.789      | 91.012         | 0.446     | 90.876
…

# Future Improvements
&nbsp; &nbsp;&nbsp; - Add weather and incident data
&nbsp; &nbsp;&nbsp; - Build an interactive dashboard (e.g. Streamlit)
&nbsp; &nbsp;&nbsp; - Track model performance across the season
