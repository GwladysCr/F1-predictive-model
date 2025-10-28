# F1 Race Pace Prediction
This project uses historical and current Formula 1 data to predict race pace for upcoming Grand Prix events. It leverages the FastF1 library to extract session data and applies machine learning models to forecast driver performance.  
Two complementary approaches are implemented:  
&dnsb; &dnsb;&dnsb; - Circuit-based analysis: Uses past editions of the same Grand Prix.  
&dnsb; &dnsb;&dnsb; - Season-based analysis: Uses recent races from the current season.  

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
&dnsb; &dnsb;&dnsb; - Qualifying performance (QualiTime_s)  
&dnsb; &dnsb;&dnsb; - Clean air pace from previous races (CleanAirPace_s)  
&dnsb; &dnsb;&dnsb; - Team performance score (TeamScore)  

# Models Used
&dnsb; &dnsb;&dnsb; - XGBoost Regressor  
&dnsb; &dnsb;&dnsb; - Random Forest Regressor  
&dnsb; &dnsb;&dnsb; - Ensemble Model: average of both predictions  

Model performance is evaluated using:  
&dnsb; &dnsb;&dnsb; - Cross-validation (MAE)  
&dnsb; &dnsb;&dnsb; - Residual analysis  
&dnsb; &dnsb;&dnsb; - Feature importance plots  

# Visual Outputs
Each script generates:  
&dnsb; &dnsb;&dnsb; - Residual plot (Residuals.png)  
&dnsb; &dnsb;&dnsb; - Feature importance plots for each model  
&dnsb; &dnsb;&dnsb; - Top 3 predicted drivers by model  

# How to Use
  1. Run a prediction:  
&dnsb; &dnsb;&dnsb; - For circuit-based analysis: python predictions_same_circuit.py
&dnsb; &dnsb;&dnsb; - For season-based analysis: python predictions_same_season.py
	2. View results:  
&dnsb; &dnsb;&dnsb; - Predictions are printed in the console.  
&dnsb; &dnsb;&dnsb; - Plots are saved automatically.  

# Utility Functions
Located in main_functions.py:
&dnsb; &dnsb;&dnsb; - calculate_clean_air_pace: computes average of the fastest 20% laps  
&dnsb; &dnsb;&dnsb; - print_residuals: visualizes prediction errors  
&dnsb; &dnsb;&dnsb; - plot_feature_importance: shows model feature weights  
&dnsb; &dnsb;&dnsb; - plot_top3_drivers: highlights top predicted performers    

# Example Output
Predicted race pace order for Singapore
Driver   | QualiTimes | CleanAirPaces | TeamScore | Ensemble_Pred
-------- |-------------|----------------|-----------|----------------
NOR      | 89.123      | 90.456         | 1.000     | 90.321
VER      | 89.789      | 91.012         | 0.446     | 90.876
…

# Future Improvements
&dnsb; &dnsb;&dnsb; - Add weather and incident data
&dnsb; &dnsb;&dnsb; - Build an interactive dashboard (e.g. Streamlit)
&dnsb; &dnsb;&dnsb; - Track model performance across the season
