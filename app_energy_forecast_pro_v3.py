"""
app_energy_forecast_pro_v3.py
Professional Streamlit app for Energy Forecasting (LSTM,  Prophet)
Features:
- Loads cleaned_energy.csv from same folder
- Auto-loads saved models from ./models/ if present
- Trains and saves models if not present
- Student Mode toggle: simplifies UI for teaching/demo
- Tabs: Overview, LSTM,  Prophet, Comparison, Exercises
- Uses Plotly for interactive charts
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta

# ML tools
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# ----------------------
# App configuration
# ----------------------
st.set_page_config(page_title="Energy Forecasting demo (v3)",
                   layout="wide", page_icon="⚡")

st.title("⚡ Energy Forecasting & Comparison — demo (v3)")
st.markdown("**Portfolio-ready project:** LSTM,  and Prophet comparison with Student Mode for teaching.")

# ----------------------
# Paths and files
# ----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "cleaned_energy.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
# LSTM_MODEL_FILE = os.path.join(MODELS_DIR, "lstm_model.h5")
LSTM_MODEL_FILE = os.path.join(MODELS_DIR, "lstm_model.keras")
ARIMA_MODEL_FILE = os.path.join(MODELS_DIR, "arima_model.pkl")
PROPHET_MODEL_FILE = os.path.join(MODELS_DIR, "prophet_model.pkl")

os.makedirs(MODELS_DIR, exist_ok=True)

# ----------------------
# Sidebar: Student Mode and quick help
# ----------------------
st.sidebar.header("⚙️ Settings")
student_mode = st.sidebar.checkbox("Student Mode (simplified UI)", value=True)
st.sidebar.markdown("Student Mode hides advanced training options for demo clarity.")

# ----------------------
# Utility functions
# ----------------------
@st.cache_data
def load_data(path=DATA_FILE):
    """Load cleaned CSV; expect a 'date' or datetime index and a numeric 'consumption' column
    If file missing, raise an informative error.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found at: {path}\nPlease place 'cleaned_energy.csv' in the app folder.")
    df = pd.read_csv(path)
    # Try to parse date column
    if 'DateTime' in df.columns:
        df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')
        df = df.dropna(subset=['DateTime']).set_index('DateTime')
    else:
        # try to parse first column as datetime
        try:
            df.iloc[:,0] = pd.to_datetime(df.iloc[:,0], errors='coerce')
            df = df.dropna(subset=[df.columns[0]]).set_index(df.columns[0])
        except Exception:
            pass
    # unify consumption column
    if 'consumption' not in df.columns:
        if 'Zone 1 Power Consumption' in df.columns:
            df['consumption'] = pd.to_numeric(df['Zone 1 Power Consumption'], errors='coerce')
        else:
            # try first numeric column
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if num_cols:
                df['consumption'] = df[num_cols[0]]
            else:
                raise ValueError("No numeric columns found in dataset to use as 'consumption'.")
    df['consumption'] = pd.to_numeric(df['consumption'], errors='coerce').ffill().bfill()
    return df

# ----------------------
# Load dataset (with error handling)
# ----------------------
try:
    df = load_data()
    st.success(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
except Exception as e:
    st.error(f"Could not load dataset: {e}")
    st.stop()

# ----------------------
# Tabs layout
# ----------------------
# tab_overview, tab_lstm, tab_arima, tab_prophet, tab_comp, tab_ex = st.tabs([
#     "📄 Overview", "🧠 LSTM", "📊 ARIMA", "🔮 Prophet", "📈 Comparison", "🧩 Exercises & README"
# ])
tab_overview, tab_lstm,  tab_prophet, tab_comp, tab_ex = st.tabs([
    "📄 Overview", "🧠 LSTM", "🔮 Prophet", "📈 Comparison", "🧩 Exercises & README"
])

# ----------------------
# Overview Tab
# ----------------------
with tab_overview:
    st.header("Project Overview & Data Snapshot")
    st.markdown("This app is part of a portfolio-quality Energy Forecasting project comparing LSTM, and Prophet models.")
    st.subheader("Data preview")
    st.dataframe(df.head(10))

    st.subheader("Interactive consumption plot")
    # date filter
    start = st.date_input("Start date", df.index.min().date())
    end = st.date_input("End date", df.index.max().date())
    if start > end:
        st.error("Start date must be before end date.")
    else:
        df_range = df.loc[str(start):str(end)]
        fig = px.line(df_range, y='consumption', title="Consumption Time Series", labels={'consumption':'Consumption'})
        st.plotly_chart(fig, use_container_width=True)

# ----------------------
# LSTM Tab
# ----------------------
with tab_lstm:
    st.header("LSTM Model — Sequence Forecasting")
    st.markdown("Train or load an LSTM model for 1-step ahead forecasting. Student Mode limits options for teaching demos.")

    # show model info if exists
    lstm_loaded = os.path.exists(LSTM_MODEL_FILE)
    st.write("LSTM model file:", LSTM_MODEL_FILE, " — exists:" , lstm_loaded)

    if student_mode:
        train_lstm = st.button("Train LSTM (quick demo)")
    else:
        train_lstm = st.button("Train LSTM (full options)")

    if train_lstm:
        # prepare sequences
        from sklearn.preprocessing import MinMaxScaler
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
        except Exception as e:
            st.error(f"TensorFlow not available: {e}\nInstall tensorflow to use LSTM.")
            train_lstm = False

        series = df['consumption'].resample('H').mean().ffill().values.reshape(-1,1)
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(series)

        seq_len = 24 if student_mode else st.number_input("Sequence length (hours)", min_value=6, max_value=168, value=24, step=6)
        Xs, ys = [], []
        for i in range(seq_len, len(scaled)):
            Xs.append(scaled[i-seq_len:i, 0])
            ys.append(scaled[i, 0])
        Xs, ys = np.array(Xs), np.array(ys)
        Xs = Xs.reshape((Xs.shape[0], Xs.shape[1], 1))
        split_idx = int(0.8 * len(Xs))
        X_train, X_test = Xs[:split_idx], Xs[split_idx:]
        y_train, y_test = ys[:split_idx], ys[split_idx:]

        # build model
        import tensorflow as tf
        tf.keras.backend.clear_session()
        model = Sequential([
            LSTM(64, return_sequences=False, input_shape=(X_train.shape[1], 1)),
            # Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        # train (small epochs when student mode)
        epochs = 5 if student_mode else st.number_input("Epochs", 1, 200, value=10)
        with st.spinner("Training LSTM..."):
            model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0)
        st.success("LSTM training complete.")

        # save model and scaler
        try:
            model.save(LSTM_MODEL_FILE)
            joblib.dump(scaler, os.path.join(MODELS_DIR, "lstm_scaler.pkl"))
            st.info(f"Saved LSTM model to {LSTM_MODEL_FILE}")
        except Exception as e:
            st.warning(f"Could not save LSTM model: {e}")

        # evaluate
        pred_scaled = model.predict(X_test).flatten()
        pred = scaler.inverse_transform(pred_scaled.reshape(-1,1)).flatten()
        true = scaler.inverse_transform(y_test.reshape(-1,1)).flatten()
        mae = mean_absolute_error(true, pred)
        rmse = mean_squared_error(true, pred)
        st.write(f"LSTM MAE: {mae:.4f}, RMSE: {rmse:.4f}")

        fig = go.Figure()
        n_plot = min(200, len(pred))
        fig.add_trace(go.Scatter(y=true[-n_plot:], name='Actual'))
        fig.add_trace(go.Scatter(y=pred[-n_plot:], name='Predicted (LSTM)'))
        st.plotly_chart(fig, use_container_width=True)

    # option to load saved model for forecasting
    if lstm_loaded and st.button("Load LSTM from models/ (for forecasting)"):
        st.success("LSTM model will be used for forecasting in Forecast tab.")

# # ----------------------
# # ARIMA Tab
# # ----------------------
# with tab_arima:
#     st.header("ARIMA Model (pmdarima auto_arima)")
#     arima_loaded = os.path.exists(ARIMA_MODEL_FILE)
#     st.write("ARIMA model file:", ARIMA_MODEL_FILE, " — exists:" , arima_loaded)

#     if student_mode:
#         run_arima = st.button("Run auto_arima (demo)")
#     else:
#         run_arima = st.button("Run auto_arima (full)")

#     if run_arima:
#         try:
#             import pmdarima as pm
#         except Exception as e:
#             st.error(f"pmdarima not available: {e}\nInstall with: pip install pmdarima")
#             run_arima = False

#         if run_arima:
#             series = df['consumption'].resample('h').mean().ffill()
#             with st.spinner("Running auto_arima..."):
#                 model = pm.auto_arima(series, seasonal=True, m=24, error_action='ignore', suppress_warnings=True)
#             st.success("ARIMA (auto_arima) complete.")
#             # save model
#             try:
#                 import joblib
#                 joblib.dump(model, ARIMA_MODEL_FILE)
#                 st.info(f"Saved ARIMA model to {ARIMA_MODEL_FILE}")
#             except Exception as e:
#                 st.warning(f"Could not save ARIMA model: {e}")

#             # forecast demo
#             n_periods = st.slider("Forecast horizon (hours)", 24, 168, 24)
#             fc, confint = model.predict(n_periods=n_periods, return_conf_int=True)
#             idx = pd.date_range(start=df.index[-1] + timedelta(hours=1), periods=n_periods, freq='H')
#             fig = go.Figure()
#             fig.add_trace(go.Scatter(x=df.index[-72:], y=df['consumption'].tail(72), mode='lines', name='Recent'))
#             fig.add_trace(go.Scatter(x=idx, y=fc, mode='lines+markers', name='ARIMA Forecast'))
#             st.plotly_chart(fig, use_container_width=True)

# ----------------------
# Prophet Tab
# ----------------------
with tab_prophet:
    st.header("Prophet Model (Facebook/Meta Prophet)")
    prophet_loaded = os.path.exists(PROPHET_MODEL_FILE)
    st.write("Prophet model file:", PROPHET_MODEL_FILE, " — exists:" , prophet_loaded)

    if student_mode:
        run_prophet = st.button("Run Prophet (demo)")
    else:
        run_prophet = st.button("Run Prophet (full)")

    if run_prophet:
        try:
            from prophet import Prophet
        except Exception as e:
            st.error(f"Prophet not available: {e}\nInstall with: pip install prophet")
            run_prophet = False

        if run_prophet:
            series = df['consumption'].resample('h').mean().ffill().reset_index().rename(columns={'DateTime':'ds','consumption':'y'})
            m = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False)
            with st.spinner("Fitting Prophet..."):
                m.fit(series[:-24*7])  # fit on all but last week
            st.success("Prophet fitted.")
            # save model if possible
            try:
                import joblib
                joblib.dump(m, PROPHET_MODEL_FILE)
                st.info(f"Saved Prophet model to {PROPHET_MODEL_FILE}")
            except Exception as e:
                st.warning(f"Could not save Prophet model: {e}")

            future = m.make_future_dataframe(periods=24*7, freq='H')
            forecast = m.predict(future)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=series['ds'], y=series['y'], mode='lines', name='Actual'))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Prophet forecast'))
            st.plotly_chart(fig, use_container_width=True)

# ----------------------
# Comparison Tab
# ----------------------
with tab_comp:
    st.header("Model Comparison (Overlay)")
    st.markdown("If you have saved models in /models, this tab will load their forecasts and compare them visually.")

    # Attempt to load forecasts from models (if exist)
    forecasts = {}
    # # ARIMA forecast loader
    # if os.path.exists(ARIMA_MODEL_FILE):
    #     try:
    #         import joblib
    #         arima_model = joblib.load(ARIMA_MODEL_FILE)
    #         # produce forecast for next 168 hours for comparison
    #         fc = arima_model.predict(n_periods=168)
    #         idx = pd.date_range(start=df.index[-1] + timedelta(hours=1), periods=168, freq='H')
    #         forecasts['ARIMA'] = (idx, fc)
    #     except Exception as e:
    #         st.warning(f"Could not load ARIMA model: {e}")

    # Prophet forecast loader
    if os.path.exists(PROPHET_MODEL_FILE):
        try:
            import joblib
            m = joblib.load(PROPHET_MODEL_FILE)
            future = m.make_future_dataframe(periods=168, freq='H')
            forecast = m.predict(future)
            forecasts['Prophet'] = (forecast['ds'].tail(168), forecast['yhat'].tail(168).values)
        except Exception as e:
            st.warning(f"Could not load Prophet model: {e}")

    # LSTM forecast loader
    if os.path.exists(LSTM_MODEL_FILE) and os.path.exists(os.path.join(MODELS_DIR, "lstm_scaler.pkl")):
        try:
            from tensorflow.keras.models import load_model
            scaler = joblib.load(os.path.join(MODELS_DIR, "lstm_scaler.pkl"))
            model = load_model(LSTM_MODEL_FILE)
            # create iterative forecast for 168 hours
            history = df['consumption'].resample('H').mean().ffill().values
            seq_len = 24
            cur_seq = scaler.transform(history[-seq_len:].reshape(-1,1)).flatten()
            preds = []
            for _ in range(168):
                x_in = cur_seq.reshape(1, seq_len, 1)
                yhat = model.predict(x_in, verbose=0)[0,0]
                preds.append(yhat)
                cur_seq = np.append(cur_seq[1:], yhat)
            preds_inv = scaler.inverse_transform(np.array(preds).reshape(-1,1)).flatten()
            idx = pd.date_range(start=df.index[-1] + timedelta(hours=1), periods=168, freq='H')
            forecasts['LSTM'] = (idx, preds_inv)
        except Exception as e:
            st.warning(f"Could not load LSTM model: {e}")

    # Plot if at least one forecast exists
    if forecasts:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index[-168:], y=df['consumption'].tail(168), mode='lines', name='Recent Actual'))
        for name, (idx, vals) in forecasts.items():
            fig.add_trace(go.Scatter(x=idx, y=vals, mode='lines', name=name + ' forecast'))
        fig.update_layout(title="Model Comparison: Next 168 hours", xaxis_title="Date", yaxis_title="Consumption")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No saved model forecasts found in /models. Train and save models in their tabs first.")

# ----------------------
# Exercises & README Tab
# ----------------------
with tab_ex:
    st.header("Project README & Exercises")
    st.markdown("This project is designed for portfolio presentation and teaching. Include the following in your portfolio page:")
    st.markdown("""
- Short description of problem and dataset
- Models: LSTM,  Prophet — comparison and insights
- Link to interactive Streamlit demo and GitHub repo with Notebook
- Screenshots and short video demo (optional)
""")
    st.subheader("Student Exercises")
    st.markdown("""
1. Add weather features and evaluate model improvement (train RF & LSTM with exogenous variables).  
2. Implement time-series cross-validation and compare results across methods.  
3. Tune LSTM hyperparameters and visualize training/validation loss curves.  
""")

st.markdown("---")
st.caption("Project generated for portfolio use. Place 'cleaned_energy.csv' and optional models in the same folder.")
