"""
01_data_preprocessing.py
VRU COMPRESSOR - DATA PREPROCESSING

Input:  vru_data_full_4years.csv
Output: vru_preprocessed.csv

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

# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_FILE  = 'vru data full 4years.csv'
OUTPUT_FILE = 'vru_preprocessed.csv'
PLOT_DIR    = 'plots/'

PHYSICAL_BOUNDS = {
    'discharge_temp':     (50, 350),
    'discharge_pressure': (2, 45),
    'jacket_water':       (2, 35),
}

OFF_THRESHOLD_TEMP = 90  # °F

# =============================================================================
# STEP 1: LOAD RAW DATA
# =============================================================================

def load_raw_data():
    print("=" * 70)
    print(" STEP 1: LOADING RAW DATA")
    print("=" * 70)

    df = pd.read_csv(INPUT_FILE, parse_dates=['date'])
    df.set_index('date', inplace=True)

    print(f"  File       : {INPUT_FILE}")
    print(f"  Rows       : {len(df)}")
    print(f"  Date range : {df.index.min().date()} to {df.index.max().date()}")
    print(f"  Columns    : {list(df.columns)}")
    print()

    sensor_cols = ['discharge_temp', 'discharge_pressure', 'jacket_water']
    meta_cols   = [c for c in ['status', 'failure_type', 'is_down', 'is_degrading'] if c in df.columns]
    df = df[sensor_cols + meta_cols]

    print("  Raw Data Statistics:")
    for col in sensor_cols:
        s = df[col]
        print(f"  {col:25s}: mean={s.mean():7.1f}  std={s.std():6.1f}  "
              f"min={s.min():7.1f}  max={s.max():7.1f}  null={s.isnull().sum()}")
    print()

    return df

# =============================================================================
# STEP 2: FILTER COMPRESSOR-OFF DAYS
# =============================================================================

def filter_off_days(df):
    print("=" * 70)
    print(" STEP 2: FILTER COMPRESSOR-OFF DAYS")
    print("=" * 70)

    n_before = len(df)

    if 'is_down' in df.columns:
        off_mask = df['is_down'] == 1
        print(f"  Down days (from status column): {off_mask.sum()}")
    else:
        off_mask = pd.Series(False, index=df.index)

    temp_off    = df['discharge_temp'] < OFF_THRESHOLD_TEMP
    combined_off = off_mask | temp_off
    df_running  = df[~combined_off].copy()

    print(f"  Days with temp < {OFF_THRESHOLD_TEMP}°F : {temp_off.sum()}")
    print(f"  Total OFF days removed         : {combined_off.sum()} ({combined_off.sum()/n_before*100:.1f}%)")
    print(f"  Remaining running days         : {len(df_running)}")
    print()

    return df_running

# =============================================================================
# STEP 3: OUTLIER DETECTION AND REMOVAL
# =============================================================================

def remove_outliers(df):
    print("=" * 70)
    print(" STEP 3: OUTLIER DETECTION AND REMOVAL")
    print("=" * 70)

    for col in ['discharge_temp', 'discharge_pressure', 'jacket_water']:
        lo, hi   = PHYSICAL_BOUNDS[col]
        outliers = (df[col] < lo) | (df[col] > hi)
        if outliers.sum() > 0:
            print(f"  {col}: {outliers.sum()} outliers outside [{lo}, {hi}] → set to NaN")
            df.loc[outliers, col] = np.nan
        else:
            print(f"  {col}: 0 outliers")
    print()

    return df

# =============================================================================
# STEP 4: GAP FILLING
# =============================================================================

def handle_missing(df):
    print("=" * 70)
    print(" STEP 4: GAP FILLING")
    print("=" * 70)

    sensor_cols = ['discharge_temp', 'discharge_pressure', 'jacket_water']
    before = df[sensor_cols].isnull().sum().sum()
    print(f"  Missing values before fill: {before}")

    df[sensor_cols] = df[sensor_cols].ffill(limit=2)
    df[sensor_cols] = df[sensor_cols].bfill(limit=2)

    after = df[sensor_cols].isnull().sum().sum()
    print(f"  Missing values after fill : {after}")

    if after > 0:
        print(f"  Dropping {after} remaining NaN rows")
        df = df.dropna(subset=sensor_cols)

    print(f"  Final dataset: {len(df)} observations")
    print()

    return df

# =============================================================================
# STEP 5: ENSURE DAILY FREQUENCY
# =============================================================================

def ensure_daily_freq(df):
    print("=" * 70)
    print(" STEP 5: ENSURE DAILY FREQUENCY")
    print("=" * 70)

    sensor_cols = ['discharge_temp', 'discharge_pressure', 'jacket_water']
    date_diff   = pd.Series(df.index).diff().dt.days
    gaps        = date_diff[date_diff > 1]

    if len(gaps) > 0:
        print(f"  Found {len(gaps)} gaps in date index")
        full_range = pd.date_range(df.index.min(), df.index.max(), freq='D')
        df = df.reindex(full_range)
        df[sensor_cols] = df[sensor_cols].ffill(limit=2)
        df = df.dropna(subset=sensor_cols)
        print(f"  After reindexing: {len(df)} observations")
    else:
        print("  No gaps found — dataset is contiguous")
        df = df.asfreq('D')
        df = df.dropna(subset=sensor_cols)

    print(f"  Final : {len(df)} daily observations")
    print(f"  Range : {df.index.min().date()} to {df.index.max().date()}")
    print()

    return df

# =============================================================================
# STEP 6: SAVE AND VISUALIZE
# =============================================================================

def save_and_plot(df):
    print("=" * 70)
    print(" STEP 6: SAVE AND VISUALIZE")
    print("=" * 70)

    os.makedirs(PLOT_DIR, exist_ok=True)

    sensor_cols = ['discharge_temp', 'discharge_pressure', 'jacket_water']
    df_out      = df[sensor_cols].copy()
    df_out.index.name = 'date'
    df_out.to_csv(OUTPUT_FILE)
    print(f"  ✓ Saved: {OUTPUT_FILE} ({len(df_out)} rows)")

    print()
    print("  Final Parameter Statistics:")
    for col in sensor_cols:
        s = df_out[col]
        print(f"  {col:25s}: mean={s.mean():7.1f}  std={s.std():5.1f}  "
              f"min={s.min():7.1f}  max={s.max():7.1f}")
    print()

    params = [
        ('discharge_temp',     'Discharge Temperature (°F)', 'tab:red',   (110, 150)),
        ('discharge_pressure', 'Discharge Pressure (psi)',   'tab:blue',  (10, 30)),
        ('jacket_water',       'Jacket Water Pressure (psi)','tab:green', (12, 20)),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    for ax, (col, label, color, (nlo, nhi)) in zip(axes, params):
        ax.plot(df_out.index, df_out[col], color=color, linewidth=0.6, alpha=0.8)
        ax.axhspan(nlo, nhi, alpha=0.1, color='green')
        ax.axhline(nlo, color='green', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.axhline(nhi, color='green', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.set_ylabel(label, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_title(label, fontsize=11)
    axes[-1].set_xlabel('Date')
    plt.suptitle('VRU Compressor — Preprocessed Daily Data (Running Days Only)', fontsize=13)
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}01_preprocessed_timeseries.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Plot saved: {PLOT_DIR}01_preprocessed_timeseries.png")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (col, label, color, _) in zip(axes, params):
        ax.hist(df_out[col], bins=30, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.set_xlabel(label.split('(')[0].strip())
        ax.set_ylabel('Count')
        ax.axvline(df_out[col].mean(), color='black', linestyle='--', linewidth=1, label=f'Mean={df_out[col].mean():.1f}')
        ax.legend(fontsize=8)
    plt.suptitle('Parameter Distributions (Running Days)', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}01_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Plot saved: {PLOT_DIR}01_distributions.png")
    print()

    return df_out

# =============================================================================
# MAIN
# =============================================================================

def main():
    print()
    print("=" * 70)
    print(" VRU COMPRESSOR — DATA PREPROCESSING")
    print(f" Input : {INPUT_FILE}")
    print(f" Output: {OUTPUT_FILE}")
    print("=" * 70)
    print()

    df = load_raw_data()
    df = filter_off_days(df)
    df = remove_outliers(df)
    df = handle_missing(df)
    df = ensure_daily_freq(df)
    save_and_plot(df)

    print("=" * 70)
    print(" PREPROCESSING COMPLETE")
    print("=" * 70)
    print(f"  {INPUT_FILE} → {OUTPUT_FILE}")
    print(f"  Next: run 02_arima_modeling.py")
    print()

if __name__ == "__main__":
    main()
