"""
02_arima_modeling.py
VRU COMPRESSOR - ARIMA TIME-SERIES FORECASTING

Input:  vru_preprocessed.csv (dari 01_data_preprocessing.py)
Output: arima_models/forecasts_*.csv, arima_models/model_metadata.json

Author: Kaa Albaraq Sakha
Thesis: Time-Series Forecasting for Preventive Maintenance of VRU Compressors
Institution: Universitas Gadjah Mada
Date: April 2026
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json
import warnings

from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import shapiro

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_FILE       = 'vru_preprocessed.csv'
OUTPUT_DIR       = 'arima_models/'
TRAIN_TEST_SPLIT = 0.85
FORECAST_HORIZON = 5

SUDDEN_EVENT_THRESHOLDS = {
    'discharge_temp':     {'spike':  30, 'drop': -30},
    'discharge_pressure': {'spike':  10, 'drop': -10},
    'jacket_water':       {'spike':   5, 'drop':  -5},
}

# =============================================================================
# STEP 1: LOAD DATA
# =============================================================================

def load_data():
    print("=" * 70)
    print("STEP 1: LOADING CLEANED DATA")
    print("=" * 70)

    df = pd.read_csv(INPUT_FILE)
    # Handle both 'date' column and unnamed index from preprocessing output
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
    else:
        df.index = pd.to_datetime(df.index)

    df = df[['discharge_temp', 'discharge_pressure', 'jacket_water']].copy()
    df = df.asfreq('D')
    df = df.ffill(limit=2)

    remaining = df.isnull().sum().sum()
    if remaining > 0:
        df = df.dropna()

    print(f"  Loaded {len(df)} observations")
    print(f"  Date range: {df.index.min().date()} to {df.index.max().date()}")
    print()

    return df

# =============================================================================
# STEP 2: SUDDEN EVENT DETECTION AND FILTERING
# =============================================================================

def detect_sudden_events(df):
    print("=" * 70)
    print("STEP 2: SUDDEN EVENT DETECTION AND FILTERING")
    print("=" * 70)

    sudden_mask = pd.Series(False, index=df.index)

    temp_change     = df['discharge_temp'].diff()
    pressure_change = df['discharge_pressure'].diff()
    jacket_change   = df['jacket_water'].diff()

    temp_sudden = (
        (temp_change > SUDDEN_EVENT_THRESHOLDS['discharge_temp']['spike']) |
        (temp_change < SUDDEN_EVENT_THRESHOLDS['discharge_temp']['drop'])
    )
    pressure_sudden = (
        (pressure_change > SUDDEN_EVENT_THRESHOLDS['discharge_pressure']['spike']) |
        (pressure_change < SUDDEN_EVENT_THRESHOLDS['discharge_pressure']['drop'])
    )
    jacket_sudden = (
        (jacket_change > SUDDEN_EVENT_THRESHOLDS['jacket_water']['spike']) |
        (jacket_change < SUDDEN_EVENT_THRESHOLDS['jacket_water']['drop'])
    )

    sudden_mask = temp_sudden | pressure_sudden | jacket_sudden

    # Flag 2-day recovery window after each sudden event
    for i in range(1, 3):
        sudden_mask = sudden_mask | sudden_mask.shift(-i).fillna(False)

    print(f"  Temperature spikes/drops : {temp_sudden.sum()} days")
    print(f"  Pressure spikes/drops    : {pressure_sudden.sum()} days")
    print(f"  Jacket water spikes/drops: {jacket_sudden.sum()} days")
    print(f"  Total filtered (incl. recovery): {sudden_mask.sum()} days ({sudden_mask.sum()/len(df)*100:.1f}%)")
    print()

    df_filtered = df[~sudden_mask].copy()

    print(f"  Original : {len(df)} observations")
    print(f"  Filtered : {len(df_filtered)} observations")
    print()

    return df_filtered, sudden_mask

# =============================================================================
# STEP 3: TRAIN / TEST SPLIT
# =============================================================================

def train_test_split(df):
    print("=" * 70)
    print("STEP 3: TRAIN/TEST SPLIT")
    print("=" * 70)

    split_idx = int(len(df) * TRAIN_TEST_SPLIT)
    train = df.iloc[:split_idx]
    test  = df.iloc[split_idx:]

    print(f"  Training : {len(train)} obs  ({train.index.min().date()} to {train.index.max().date()})")
    print(f"  Testing  : {len(test)} obs  ({test.index.min().date()} to {test.index.max().date()})")
    print()

    return train, test

# =============================================================================
# STEP 4: STATIONARITY TEST
# =============================================================================

def test_stationarity(series, param_name):
    print(f"\n--- Stationarity Test: {param_name} ---")

    result = adfuller(series.dropna())
    print(f"  ADF Statistic : {result[0]:.4f}")
    print(f"  p-value       : {result[1]:.6f}")

    if result[1] < 0.05:
        print(f"  ✓ STATIONARY (p={result[1]:.4f} < 0.05)  →  d=0")
        return 0
    else:
        print(f"  ✗ NON-STATIONARY  →  applying first differencing  →  d=1")
        diff_result = adfuller(series.diff().dropna())
        print(f"  After differencing: p={diff_result[1]:.6f}")
        return 1

# =============================================================================
# STEP 5: ACF / PACF PLOTS
# =============================================================================

def plot_acf_pacf(series, param_name, d_order):
    series_plot = series.diff().dropna() if d_order == 1 else series.dropna()
    suffix      = " (differenced)" if d_order == 1 else ""

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    plot_acf(series_plot,  lags=20, ax=axes[0])
    plot_pacf(series_plot, lags=20, ax=axes[1])
    axes[0].set_title(f'ACF: {param_name}{suffix}')
    axes[1].set_title(f'PACF: {param_name}{suffix}')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}acf_pacf_{param_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ ACF/PACF plot saved")

# =============================================================================
# STEP 6: MODEL SELECTION VIA GRID SEARCH
# =============================================================================

def select_best_arima(train_series, param_name, d_order):
    print(f"\n--- Model Selection: {param_name} ---")
    print(f"  Testing ARIMA(p,{d_order},q) for p,q in 0..3 ...")

    best_aic   = np.inf
    best_order = None
    best_model = None
    results    = []

    for p in range(4):
        for q in range(4):
            try:
                m   = ARIMA(train_series, order=(p, d_order, q)).fit()
                results.append({'p': p, 'q': q, 'AIC': m.aic, 'BIC': m.bic})
                if m.aic < best_aic:
                    best_aic, best_order, best_model = m.aic, (p, d_order, q), m
            except Exception:
                continue

    results_df = pd.DataFrame(results).sort_values('AIC')
    print("\n  Top 5 models by AIC:")
    for _, row in results_df.head().iterrows():
        print(f"    ARIMA({int(row['p'])},{d_order},{int(row['q'])})  AIC={row['AIC']:.2f}  BIC={row['BIC']:.2f}")

    print(f"\n  ✓ Best: ARIMA{best_order}  AIC={best_aic:.2f}")
    return best_model, best_order

# =============================================================================
# STEP 7: DIAGNOSTIC VALIDATION
# =============================================================================

def validate_model(model, param_name):
    print(f"\n--- Diagnostics: {param_name} ---")

    residuals = model.resid

    lb_p = acorr_ljungbox(residuals, lags=[10], return_df=True)['lb_pvalue'].iloc[0]
    sw_p = shapiro(residuals)[1]

    print(f"  Ljung-Box p={lb_p:.4f}  {'✓ PASS' if lb_p >= 0.05 else '⚠ WARN — autocorrelation remains'}")
    print(f"  Shapiro-Wilk p={sw_p:.4f}  {'✓ PASS' if sw_p >= 0.05 else 'NOTE — affects intervals only, not point forecasts'}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0, 0].plot(residuals)
    axes[0, 0].axhline(0, color='r', linestyle='--', linewidth=0.8)
    axes[0, 0].set_title('Residuals Over Time')
    axes[0, 1].hist(residuals, bins=20, edgecolor='black', alpha=0.7)
    axes[0, 1].set_title('Residual Distribution')
    plot_acf(residuals, lags=20, ax=axes[1, 0])
    axes[1, 0].set_title('ACF of Residuals')
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}diagnostics_{param_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Diagnostic plots saved")

    return lb_p >= 0.05

# =============================================================================
# STEP 8: ROLLING WINDOW FORECAST
# =============================================================================

def generate_forecasts(train_series, test_series, best_order, param_name):
    print(f"\n--- Rolling Forecasts: {param_name} ---")
    print(f"  Method : expanding window, refit every 20 steps")
    print(f"  Horizon: {FORECAST_HORIZON} days")

    history      = list(train_series.values)
    n_forecasts  = len(test_series) - FORECAST_HORIZON + 1
    forecasts    = []
    current_model = None
    naive_errors = []

    for i in range(n_forecasts):
        if i % 20 == 0:
            try:
                current_model = ARIMA(history, order=best_order).fit()
            except Exception:
                pass

        fc = current_model.forecast(steps=FORECAST_HORIZON)

        forecasts.append({
            'forecast_date': str(test_series.index[i].date()),
            'target_date':   str(test_series.index[i + FORECAST_HORIZON - 1].date()),
            'forecast_1day': float(fc[0]),
            'forecast_5day': float(fc[FORECAST_HORIZON - 1]),
            'actual_1day':   float(test_series.iloc[i]),
            'actual_5day':   float(test_series.iloc[i + FORECAST_HORIZON - 1]),
        })

        if i > 0:
            naive_errors.append(abs(test_series.iloc[i] - test_series.iloc[i - 1]))

        history.append(test_series.iloc[i])

        if (i + 1) % 50 == 0 or i == n_forecasts - 1:
            print(f"  Progress: {i+1}/{n_forecasts}")

    forecast_df = pd.DataFrame(forecasts)
    naive_mae   = float(np.mean(naive_errors)) if naive_errors else 1.0

    print(f"  ✓ {len(forecast_df)} forecast sets generated")
    print(f"  Naive MAE (for MASE): {naive_mae:.4f}")

    return forecast_df, naive_mae

# =============================================================================
# MAIN
# =============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print()
    print("=" * 70)
    print(" VRU COMPRESSOR — ARIMA FORECASTING")
    print(" Rolling Window Forecast")
    print("=" * 70)
    print()

    df = load_data()
    df_filtered, sudden_mask = detect_sudden_events(df)
    train, test = train_test_split(df_filtered)

    parameters = {
        'discharge_temp':     'Discharge Temperature',
        'discharge_pressure': 'Discharge Pressure',
        'jacket_water':       'Jacket Water Pressure',
    }

    models_meta  = {}
    forecasts_all = {}
    naive_maes   = {}

    for param_col, param_name in parameters.items():
        print("\n" + "=" * 70)
        print(f" PARAMETER: {param_name.upper()}")
        print("=" * 70)

        train_s = train[param_col]
        test_s  = test[param_col]

        d_order = test_stationarity(train_s, param_name)
        plot_acf_pacf(train_s, param_col, d_order)
        best_model, best_order = select_best_arima(train_s, param_name, d_order)
        validate_model(best_model, param_col)
        forecast_df, naive_mae = generate_forecasts(train_s, test_s, best_order, param_name)

        models_meta[param_col] = {
            'order': best_order,
            'AIC':   float(best_model.aic),
            'BIC':   float(best_model.bic),
        }
        forecasts_all[param_col] = forecast_df
        naive_maes[param_col]    = naive_mae

        # Save model summary
        with open(f'{OUTPUT_DIR}model_summary_{param_col}.txt', 'w') as f:
            f.write(str(best_model.summary()))
        print(f"  ✓ Model summary saved")

    # Save all forecast CSVs
    for param_col, fc_df in forecasts_all.items():
        fc_df.to_csv(f'{OUTPUT_DIR}forecasts_{param_col}.csv', index=False)
        print(f"  ✓ Forecasts saved: forecasts_{param_col}.csv")

    # Save metadata
    metadata = {
        'train_size':              len(train),
        'test_size':               len(test),
        'forecast_horizon':        FORECAST_HORIZON,
        'sudden_events_filtered':  int(sudden_mask.sum()),
        'forecast_method':         'expanding_window_periodic_refit',
        'refit_interval':          20,
        'models':                  models_meta,
        'naive_mae':               naive_maes,
    }
    with open(f'{OUTPUT_DIR}model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓ Metadata saved: model_metadata.json")

    print()
    print("=" * 70)
    print(" ARIMA MODELING COMPLETE")
    print("=" * 70)
    print(f"  Next: run 03_alert_system.py")
    print()

if __name__ == "__main__":
    main()
