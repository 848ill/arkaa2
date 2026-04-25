"""
03_alert_system.py
VRU COMPRESSOR - RULE-BASED ALERT CLASSIFICATION

Input:  arima_models/forecasts_*.csv (dari 02_arima_modeling.py)
Output: alerts/alerts_generated.csv

Classification Rules:
  OFF    : Discharge temp < 90°F
  GREEN  : 0 violations
  YELLOW : 1 violation
  RED    : 2+ violations ATAU temp > 300°F (critical)

Thresholds (Ro-Flo specs + 52 bulan data operasional + 28 kegagalan tercatat):
  Discharge Temp    : Normal 110-150°F | Warning >150°F | Critical >300°F | OFF <90°F
  Discharge Pressure: Normal 10-30 psi
  Jacket Water      : Normal 12-20 psi (leading indicator, lag 7-14 hari ke temp)

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
import json
import os

# =============================================================================
# THRESHOLDS
# =============================================================================

THRESHOLDS = {
    'discharge_temp': {
        'OFF':      90,
        'NORMAL':   (110, 150),
        'WARNING':  150,
        'CRITICAL': 300,
    },
    'discharge_pressure': {
        'NORMAL': (10, 30),
        'LOW':    10,
        'HIGH':   30,
    },
    'jacket_water': {
        'NORMAL': (12, 20),
        'LOW':    12,
        'HIGH':   20,
    },
}

INPUT_DIR  = 'arima_models/'
OUTPUT_DIR = 'alerts/'

# =============================================================================
# PARAMETER CHECKS
# =============================================================================

def check_discharge_temp(value):
    if value < THRESHOLDS['discharge_temp']['OFF']:
        return False, False, 'OFF'
    if value > THRESHOLDS['discharge_temp']['CRITICAL']:
        return True, True, f'CRITICAL: Temp {value:.1f}°F (>{THRESHOLDS["discharge_temp"]["CRITICAL"]}°F)'
    if value > THRESHOLDS['discharge_temp']['WARNING']:
        return True, False, f'Temp HIGH {value:.1f}°F (>{THRESHOLDS["discharge_temp"]["WARNING"]}°F)'
    lo, hi = THRESHOLDS['discharge_temp']['NORMAL']
    if lo <= value <= hi:
        return False, False, f'Temp OK {value:.1f}°F'
    return False, False, f'Temp {value:.1f}°F (startup/warmup zone)'

def check_discharge_pressure(value):
    lo, hi = THRESHOLDS['discharge_pressure']['NORMAL']
    if value < THRESHOLDS['discharge_pressure']['LOW']:
        return True, False, f'Press LOW {value:.1f} psi (<{lo} psi — vane wear / internal leakage)'
    if value > THRESHOLDS['discharge_pressure']['HIGH']:
        return True, False, f'Press HIGH {value:.1f} psi (>{hi} psi — downstream restriction)'
    return False, False, f'Press OK {value:.1f} psi'

def check_jacket_water(value):
    """
    Jacket water is a LEADING INDICATOR.
    Pressure decline here typically precedes temperature rise by 7-14 days
    due to thermal lag in the cooling system.
    """
    lo, hi = THRESHOLDS['jacket_water']['NORMAL']
    if value < THRESHOLDS['jacket_water']['LOW']:
        return True, False, f'Jacket LOW {value:.1f} psi (<{lo} psi — cooling degradation)'
    if value > THRESHOLDS['jacket_water']['HIGH']:
        return True, False, f'Jacket HIGH {value:.1f} psi (>{hi} psi — flow restriction)'
    return False, False, f'Jacket OK {value:.1f} psi'

# =============================================================================
# CORE CLASSIFICATION
# =============================================================================

def classify_alert(temp_value, pressure_value, jacket_value):
    """
    Three-step classification:
    1. Check if compressor is running (OFF gate)
    2. Check each parameter for violations
    3. Count violations and classify
    """

    # Step 1: OFF check
    if temp_value < THRESHOLDS['discharge_temp']['OFF']:
        return {
            'status':      'OFF',
            'violations':  [],
            'n_violations': 0,
            'is_critical': False,
            'details':     {'discharge_temp': 'OFF', 'discharge_pressure': 'N/A', 'jacket_water': 'N/A'},
            'reason':      f'Compressor not running (temp {temp_value:.1f}°F < {THRESHOLDS["discharge_temp"]["OFF"]}°F)',
        }

    # Step 2: Check each parameter
    temp_viol,   temp_crit,   temp_desc   = check_discharge_temp(temp_value)
    press_viol,  press_crit,  press_desc  = check_discharge_pressure(pressure_value)
    jacket_viol, jacket_crit, jacket_desc = check_jacket_water(jacket_value)

    violations  = []
    is_critical = False

    if temp_viol:
        violations.append(temp_desc)
        if temp_crit:
            is_critical = True

    if press_viol:
        violations.append(press_desc)

    if jacket_viol:
        violations.append(jacket_desc)

    n_violations = len(violations)

    # Step 3: Classify
    if is_critical or n_violations >= 2:
        status = 'RED'
        reason = f'CRITICAL violation' if is_critical else f'{n_violations} parameters out of range'
    elif n_violations == 1:
        status = 'YELLOW'
        reason = violations[0]
    else:
        status = 'GREEN'
        reason = 'All parameters within normal operating ranges'

    details = {
        'discharge_temp':     'CRITICAL' if temp_crit else ('VIOLATION' if temp_viol else 'OK'),
        'discharge_pressure': 'VIOLATION' if press_viol else 'OK',
        'jacket_water':       'VIOLATION' if jacket_viol else 'OK',
    }

    return {
        'status':       status,
        'violations':   violations,
        'n_violations': n_violations,
        'is_critical':  is_critical,
        'details':      details,
        'reason':       reason,
    }

# =============================================================================
# GENERATE ALERTS
# =============================================================================

def generate_alerts():
    print("=" * 70)
    print(" VRU COMPRESSOR — ALERT CLASSIFICATION SYSTEM")
    print(" OFF / GREEN / YELLOW / RED")
    print("=" * 70)
    print()

    # Load forecasts
    forecasts = {}
    for param in ['discharge_temp', 'discharge_pressure', 'jacket_water']:
        fpath = f'{INPUT_DIR}forecasts_{param}.csv'
        if not os.path.exists(fpath):
            print(f"  ERROR: {fpath} not found — run 02_arima_modeling.py first")
            return None
        forecasts[param] = pd.read_csv(fpath)

    n_rows = min(len(forecasts[p]) for p in forecasts)
    print(f"  Loaded {n_rows} forecast points per parameter")
    print()

    # Print thresholds
    print("  Thresholds:")
    print(f"  Discharge Temp : Normal {THRESHOLDS['discharge_temp']['NORMAL']}°F | Warning >{THRESHOLDS['discharge_temp']['WARNING']}°F | Critical >{THRESHOLDS['discharge_temp']['CRITICAL']}°F")
    print(f"  Discharge Press: Normal {THRESHOLDS['discharge_pressure']['NORMAL']} psi")
    print(f"  Jacket Water   : Normal {THRESHOLDS['jacket_water']['NORMAL']} psi")
    print()

    alert_results = []

    for i in range(n_rows):
        temp_5d   = forecasts['discharge_temp']['forecast_5day'].iloc[i]
        press_5d  = forecasts['discharge_pressure']['forecast_5day'].iloc[i]
        jacket_5d = forecasts['jacket_water']['forecast_5day'].iloc[i]

        temp_1d   = forecasts['discharge_temp']['forecast_1day'].iloc[i]
        press_1d  = forecasts['discharge_pressure']['forecast_1day'].iloc[i]
        jacket_1d = forecasts['jacket_water']['forecast_1day'].iloc[i]

        temp_act  = forecasts['discharge_temp']['actual_5day'].iloc[i]
        press_act = forecasts['discharge_pressure']['actual_5day'].iloc[i]
        jacket_act = forecasts['jacket_water']['actual_5day'].iloc[i]

        res_5d  = classify_alert(temp_5d,  press_5d,  jacket_5d)
        res_1d  = classify_alert(temp_1d,  press_1d,  jacket_1d)
        res_act = classify_alert(temp_act, press_act, jacket_act)

        alert_results.append({
            'forecast_date':   forecasts['discharge_temp']['forecast_date'].iloc[i],
            'target_date':     forecasts['discharge_temp']['target_date'].iloc[i],
            'alert_status':    res_5d['status'],
            'alert_reason':    res_5d['reason'],
            'n_violations':    res_5d['n_violations'],
            'is_critical':     res_5d['is_critical'],
            'temp_status':     res_5d['details']['discharge_temp'],
            'pressure_status': res_5d['details']['discharge_pressure'],
            'jacket_status':   res_5d['details']['jacket_water'],
            'alert_1day':      res_1d['status'],
            'alert_actual':    res_act['status'],
            'temp_forecast_5d':   round(temp_5d,   2),
            'press_forecast_5d':  round(press_5d,  2),
            'jacket_forecast_5d': round(jacket_5d, 2),
            'temp_actual':     round(temp_act,  2),
            'press_actual':    round(press_act, 2),
            'jacket_actual':   round(jacket_act, 2),
            'violations':      '; '.join(res_5d['violations']) if res_5d['violations'] else '',
        })

    alerts_df = pd.DataFrame(alert_results)
    total     = len(alerts_df)

    # Print distribution
    print("  Alert Distribution (5-day forecast):")
    print("  " + "-" * 50)
    for status in ['GREEN', 'YELLOW', 'RED', 'OFF']:
        count = (alerts_df['alert_status'] == status).sum()
        pct   = count / total * 100 if total > 0 else 0
        bar   = '█' * int(pct / 2)
        print(f"  {status:6s}: {count:4d} ({pct:5.1f}%) {bar}")
    print()

    print("  Alert Distribution (actual values — ground truth):")
    print("  " + "-" * 50)
    for status in ['GREEN', 'YELLOW', 'RED', 'OFF']:
        count = (alerts_df['alert_actual'] == status).sum()
        pct   = count / total * 100 if total > 0 else 0
        print(f"  {status:6s}: {count:4d} ({pct:5.1f}%)")
    print()

    agreement     = (alerts_df['alert_status'] == alerts_df['alert_actual']).sum()
    agreement_pct = agreement / total * 100 if total > 0 else 0
    print(f"  Forecast-Actual Agreement: {agreement}/{total} ({agreement_pct:.1f}%)")
    print()

    return alerts_df

# =============================================================================
# CONFUSION MATRIX
# =============================================================================

def plot_confusion_matrix(alerts_df):
    statuses = ['OFF', 'GREEN', 'YELLOW', 'RED']

    print()
    print("  Forecast vs Actual Alert Confusion Matrix:")
    print("  " + "-" * 55)
    header = f"  {'':14s}" + "".join(f"{'Act:'+s:>10s}" for s in statuses)
    print(header)
    print("  " + "-" * 55)

    for fc_status in statuses:
        row = f"  {'Fc:'+fc_status:14s}"
        for act_status in statuses:
            count = ((alerts_df['alert_status'] == fc_status) &
                     (alerts_df['alert_actual']  == act_status)).sum()
            row += f"{count:>10d}"
        print(row)
    print("  " + "-" * 55)

# =============================================================================
# ALERT TIMELINE PLOT
# =============================================================================

def plot_alert_timeline(alerts_df):
    alerts_df = alerts_df.copy()
    alerts_df['forecast_dt'] = pd.to_datetime(alerts_df['forecast_date'])

    status_map = {'OFF': -1, 'GREEN': 0, 'YELLOW': 1, 'RED': 2}
    colors_map = {'OFF': '#9CA3AF', 'GREEN': '#16A34A', 'YELLOW': '#EAB308', 'RED': '#DC2626'}

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [2, 1, 1, 1]})

    ax = axes[0]
    alerts_df['status_num'] = alerts_df['alert_status'].map(status_map)
    for status, color in colors_map.items():
        mask = alerts_df['alert_status'] == status
        if mask.any():
            ax.scatter(alerts_df.loc[mask, 'forecast_dt'], alerts_df.loc[mask, 'status_num'],
                       c=color, s=60, alpha=0.8, label=status, edgecolors='white', linewidth=0.5)
    ax.set_yticks([-1, 0, 1, 2])
    ax.set_yticklabels(['OFF', 'GREEN', 'YELLOW', 'RED'])
    ax.set_title('VRU Compressor Alert Classification Timeline', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    params_plot = [
        ('temp_forecast_5d',  'temp_actual',   THRESHOLDS['discharge_temp']['WARNING'],   'tab:red',   'Discharge Temp (°F)'),
        ('press_forecast_5d', 'press_actual',  None,                                      'tab:blue',  'Discharge Pressure (psi)'),
        ('jacket_forecast_5d','jacket_actual', THRESHOLDS['jacket_water']['LOW'],          'tab:green', 'Jacket Water (psi)'),
    ]
    for ax, (fc_col, act_col, threshold, color, label) in zip(axes[1:], params_plot):
        ax.plot(alerts_df['forecast_dt'], alerts_df[fc_col],  color=color, linewidth=1, label='Forecast')
        ax.plot(alerts_df['forecast_dt'], alerts_df[act_col], color='black', linewidth=0.8, alpha=0.5, label='Actual')
        if threshold:
            ax.axhline(threshold, color='red', linestyle='--', linewidth=0.8, alpha=0.7, label=f'Threshold ({threshold})')
        ax.set_ylabel(label, fontsize=9)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Date')
    plt.tight_layout()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(f'{OUTPUT_DIR}alert_timeline.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  ✓ Alert timeline plot saved: {OUTPUT_DIR}alert_timeline.png")

# =============================================================================
# MAIN
# =============================================================================

def main():
    print()

    alerts_df = generate_alerts()

    if alerts_df is None:
        print("  Exiting — no forecast data available")
        return

    plot_alert_timeline(alerts_df)
    plot_confusion_matrix(alerts_df)

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    alerts_df.to_csv(f'{OUTPUT_DIR}alerts_generated.csv', index=False)
    print(f"\n  ✓ Alerts saved: {OUTPUT_DIR}alerts_generated.csv")

    # Save threshold config
    config = {}
    for param, thr in THRESHOLDS.items():
        config[param] = {k: list(v) if isinstance(v, tuple) else v for k, v in thr.items()}
    with open(f'{OUTPUT_DIR}threshold_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    print(f"  ✓ Threshold config saved: {OUTPUT_DIR}threshold_config.json")

    print()
    print("=" * 70)
    print(" ALERT CLASSIFICATION COMPLETE")
    print("=" * 70)
    print("  Next: run 04_performance_evaluation.py")
    print()

if __name__ == "__main__":
    main()
