# Energy Forecasting & Analysis — Portfolio Project (Project 1)

## Overview
This project demonstrates an end-to-end energy forecasting workflow using Machine Learning and Deep Learning techniques. It is designed to be portfolio-ready for freelance demonstrations in AI for Energy systems.

## Contents
- `app_energy_forecast_pro_v3.py` — Streamlit app for interactive exploration and forecasting.
- `Energy_Forecasting_LSTM_Comparisons.ipynb` — Jupyter Notebook with full experiments and comparisons (LSTM, RandomForest, ARIMA, Prophet).
- `cleaned_energy.csv` — Expected cleaned dataset (place in same folder).
- `models/` — Folder where trained models are saved (`lstm_model.h5`, `arima_model.pkl`, `prophet_model.pkl`).

## How to use
1. Place your cleaned dataset `cleaned_energy.csv` in this folder. The file should contain either a `date` column or a datetime-like first column, and a numeric consumption column (named `consumption` or `Global_active_power`).
2. Create a Python environment and install dependencies:
   ```bash
   pip install -r requirements.txt
   # or individually:
   pip install streamlit pandas numpy plotly scikit-learn joblib
   pip install tensorflow   # for LSTM (optional)
   pip install pmdarima     # for ARIMA (optional)
   pip install prophet      # for Prophet (optional)
