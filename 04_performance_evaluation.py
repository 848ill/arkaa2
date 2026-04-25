import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json

# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_DIR_MODELS = 'arima_models/'
INPUT_DIR_ALERTS = 'alerts/'
FAILURE_LOG_FILE = 'vru_failure_log.csv'
OUTPUT_DIR = 'performance_results/'

# Performance targets
TARGETS = {
    'MAE_temp': 6.0,          # < 6°F
    'MAE_pressure': 1.5,      # < 1.5 psi
    'MAE_jacket': 0.8,        # < 0.8 psi
    'RMSE_temp': 8.0,         # < 8°F
    'RMSE_pressure': 2.0,     # < 2 psi
    'RMSE_jacket': 1.0,       # < 1 psi
    'MAPE_1day': 10.0,        # < 10%
    'MAPE_5day': 15.0,        # < 15%
    'MASE': 1.0,              # < 1.0 (beat naive)
    'Lead_Time_Avg': 3.0,     # >= 3 days
}

PARAM_UNITS = {
    'discharge_temp': '°F',
    'discharge_pressure': 'psi',
    'jacket_water': 'psi',
}

PARAM_LABELS = {
    'discharge_temp': 'Discharge Temperature',
    'discharge_pressure': 'Discharge Pressure',
    'jacket_water': 'Jacket Water Pressure',
}

ALERT_CLASSES = ['OFF', 'GREEN', 'YELLOW', 'RED']

# =============================================================================
# SECTION 4.1: FORECAST ACCURACY
# =============================================================================

def safe_mape(actual, forecast):
    """MAPE with zero-value protection."""
    a = np.array(actual, dtype=float)
    f = np.array(forecast, dtype=float)
    mask = np.abs(a) > 1e-6
    n_excluded = (~mask).sum()
    if mask.sum() == 0:
        return np.nan, n_excluded
    mape = np.mean(np.abs((a[mask] - f[mask]) / a[mask])) * 100
    return mape, n_excluded


def evaluate_forecast_accuracy():
    print("="*70)
    print(" SECTION 4.1: FORECAST ACCURACY")
    print(" MAE / RMSE / MAPE / MASE  x  3 parameters  x  2 horizons")
    print("="*70)
    print()

    meta_path = f'{INPUT_DIR_MODELS}model_metadata.json'
    naive_maes = {}
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        naive_maes = meta.get('naive_mae', {})

    results = {}

    for param in ['discharge_temp', 'discharge_pressure', 'jacket_water']:
        label = PARAM_LABELS[param]
        unit = PARAM_UNITS[param]

        filepath = f'{INPUT_DIR_MODELS}forecasts_{param}.csv'
        if not os.path.exists(filepath):
            print(f"  WARNING: {filepath} not found — skipping {label}")
            continue

        df = pd.read_csv(filepath)

        actual_1d = df['actual_1day'].values
        pred_1d   = df['forecast_1day'].values
        actual_5d = df['actual_5day'].values
        pred_5d   = df['forecast_5day'].values

        mae_1d  = np.mean(np.abs(actual_1d - pred_1d))
        mae_5d  = np.mean(np.abs(actual_5d - pred_5d))
        rmse_1d = np.sqrt(np.mean((actual_1d - pred_1d)**2))
        rmse_5d = np.sqrt(np.mean((actual_5d - pred_5d)**2))

        mape_1d, exc_1d = safe_mape(actual_1d, pred_1d)
        mape_5d, exc_5d = safe_mape(actual_5d, pred_5d)

        naive_mae = naive_maes.get(param, None)
        if naive_mae is None or naive_mae == 0:
            ne = [abs(actual_1d[i] - actual_1d[i-1]) for i in range(1, len(actual_1d))]
            naive_mae = np.mean(ne) if len(ne) > 0 else 1.0

        mase_1d = mae_1d / naive_mae if naive_mae > 0 else np.nan
        mase_5d = mae_5d / naive_mae if naive_mae > 0 else np.nan

        print(f"  --- {label} ({unit}) ---")
        print(f"    {'Metric':<8} {'1-Day':>10} {'5-Day':>10}")
        print(f"    {'─'*30}")
        print(f"    {'MAE':<8} {mae_1d:>9.3f} {mae_5d:>9.3f}  {unit}")
        print(f"    {'RMSE':<8} {rmse_1d:>9.3f} {rmse_5d:>9.3f}  {unit}")
        print(f"    {'MAPE':<8} {mape_1d:>8.2f}% {mape_5d:>8.2f}%")
        print(f"    {'MASE':<8} {mase_1d:>9.3f} {mase_5d:>9.3f}  {'✓ beats naive' if mase_1d < 1 else '✗ below naive'}")
        print(f"    {'─'*30}")
        print(f"    Naive MAE baseline: {naive_mae:.4f} {unit}")
        if exc_1d > 0:
            print(f"    Note: {exc_1d} zero-value obs excluded from MAPE")
        print()

        results[param] = {
            'MAE_1day':   round(mae_1d, 4),
            'MAE_5day':   round(mae_5d, 4),
            'RMSE_1day':  round(rmse_1d, 4),
            'RMSE_5day':  round(rmse_5d, 4),
            'MAPE_1day':  round(mape_1d, 2) if not np.isnan(mape_1d) else None,
            'MAPE_5day':  round(mape_5d, 2) if not np.isnan(mape_5d) else None,
            'MASE_1day':  round(mase_1d, 3),
            'MASE_5day':  round(mase_5d, 3),
            'naive_mae':  round(naive_mae, 4),
            'n_forecasts': len(df),
        }

    # Summary table
    if results:
        print("  " + "="*70)
        print("  SUMMARY TABLE 4.1")
        print("  " + "─"*70)
        print(f"  {'Parameter':<22} {'MAE':>6} {'RMSE':>6} {'MAPE':>6} {'MASE':>6}  | {'MAE':>6} {'RMSE':>6} {'MAPE':>6} {'MASE':>6}")
        print(f"  {'':22s} {'── 1-Day ──':^27s}  | {'── 5-Day ──':^27s}")
        print("  " + "─"*70)
        for param in ['discharge_temp', 'discharge_pressure', 'jacket_water']:
            if param not in results: continue
            r = results[param]
            name = param.replace('_', ' ').title()[:21]
            print(f"  {name:<22} "
                  f"{r['MAE_1day']:>5.3f} {r['RMSE_1day']:>5.3f} {r['MAPE_1day']:>4.1f}% {r['MASE_1day']:>5.3f}"
                  f"  | {r['MAE_5day']:>5.3f} {r['RMSE_5day']:>5.3f} {r['MAPE_5day']:>4.1f}% {r['MASE_5day']:>5.3f}")
        avg = lambda k: np.mean([results[p][k] for p in results if results[p].get(k) is not None])
        print("  " + "─"*70)
        print(f"  {'Average':<22} "
              f"{avg('MAE_1day'):>5.3f} {avg('RMSE_1day'):>5.3f} {avg('MAPE_1day'):>4.1f}% {avg('MASE_1day'):>5.3f}"
              f"  | {avg('MAE_5day'):>5.3f} {avg('RMSE_5day'):>5.3f} {avg('MAPE_5day'):>4.1f}% {avg('MASE_5day'):>5.3f}")
        print("  " + "─"*70)
        print()

    return results


# =============================================================================
# SECTION 4.2: ALERT CLASSIFICATION PERFORMANCE
# =============================================================================

def evaluate_alert_classification():
    print("="*70)
    print(" SECTION 4.2: ALERT CLASSIFICATION PERFORMANCE")
    print(" Precision / Recall / F1  per class")
    print("="*70)
    print()

    alerts_path = f'{INPUT_DIR_ALERTS}alerts_generated.csv'
    if not os.path.exists(alerts_path):
        print(f"  ERROR: {alerts_path} not found — run 03_alert_system.py first")
        return None

    df = pd.read_csv(alerts_path)
    y_pred  = df['alert_status'].values
    y_true  = df['alert_actual'].values
    n_total = len(df)

    print(f"  Total forecast points: {n_total}")
    print()

    # Confusion matrix
    print("  CONFUSION MATRIX (rows=forecast, cols=actual):")
    print("  " + "─"*55)
    print(f"  {'Forecast / Actual':>22}" + "".join(f"{'Act:'+c:>10}" for c in ALERT_CLASSES) + f"{'Total':>8}")
    print("  " + "─"*55)

    confusion = {}
    for fc_cls in ALERT_CLASSES:
        confusion[fc_cls] = {}
        row_str   = f"  {'Fc:'+fc_cls:>22}"
        row_total = 0
        for act_cls in ALERT_CLASSES:
            count = int(((y_pred == fc_cls) & (y_true == act_cls)).sum())
            confusion[fc_cls][act_cls] = count
            row_str += f"{count:>10}"
            row_total += count
        row_str += f"{row_total:>8}"
        print(row_str)

    col_totals = f"  {'Total':>22}"
    for act_cls in ALERT_CLASSES:
        ct = sum(confusion[fc][act_cls] for fc in ALERT_CLASSES)
        col_totals += f"{ct:>10}"
    col_totals += f"{n_total:>8}"
    print("  " + "─"*55)
    print(col_totals)
    print()

    # Per-class metrics
    print("  PER-CLASS METRICS:")
    print("  " + "─"*60)
    print(f"  {'Class':<10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10} {'Predicted':>10}")
    print("  " + "─"*60)

    class_metrics = {}
    for cls in ALERT_CLASSES:
        tp      = confusion[cls].get(cls, 0)
        fp      = sum(confusion[cls].get(o, 0) for o in ALERT_CLASSES if o != cls)
        fn      = sum(confusion[fc].get(cls, 0) for fc in ALERT_CLASSES if fc != cls)
        support = tp + fn
        prec    = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0.0
        rec     = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0.0
        f1      = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        print(f"  {cls:<10} {prec:>9.1f}% {rec:>9.1f}% {f1:>9.1f}% {support:>10} {tp+fp:>10}")
        class_metrics[cls] = {'precision': round(prec, 1), 'recall': round(rec, 1),
                              'f1': round(f1, 1), 'tp': tp, 'fp': fp, 'fn': fn,
                              'support': support, 'predicted': tp+fp}

    correct  = sum(confusion[c].get(c, 0) for c in ALERT_CLASSES)
    accuracy = correct / n_total * 100 if n_total > 0 else 0
    print("  " + "─"*60)
    print(f"  {'Accuracy':<10} {accuracy:>41.1f}%   ({correct}/{n_total})")
    print()

    # Interpretation
    red    = class_metrics.get('RED', {})
    yellow = class_metrics.get('YELLOW', {})
    print("  INTERPRETATION:")
    if red.get('support', 0) > 0 and red.get('predicted', 0) == 0:
        print(f"    RED: {red['support']} actual RED days — none detected by forecast")
        print(f"    WARNING: dangerous conditions went undetected")
    if yellow.get('support', 0) > 0 and yellow.get('predicted', 0) == 0:
        print(f"    YELLOW: {yellow['support']} actual YELLOW days — none detected by forecast")
    print(f"    Root cause: ARIMA forecast anchors to historical mean (~143-146°F for temp)")
    print(f"    Actual violations require drift only 1-4°F above threshold — model not sensitive enough")
    print()

    return {'confusion_matrix': confusion, 'class_metrics': class_metrics,
            'accuracy': round(accuracy, 1), 'n_total': n_total}


# =============================================================================
# SECTION 4.2b: LEAD TIME ANALYSIS
# =============================================================================

def evaluate_lead_time():
    print("="*70)
    print(" SECTION 4.2b: LEAD TIME ANALYSIS")
    print(" Days before actual RED event, forecast already showed warning")
    print("="*70)
    print()

    alerts_path = f'{INPUT_DIR_ALERTS}alerts_generated.csv'
    if not os.path.exists(alerts_path):
        print(f"  ERROR: {alerts_path} not found")
        return None

    df = pd.read_csv(alerts_path)
    df['forecast_date'] = pd.to_datetime(df['forecast_date'])
    df['target_date']   = pd.to_datetime(df['target_date'])

    n_red    = (df['alert_actual'] == 'RED').sum()
    n_yellow = (df['alert_actual'] == 'YELLOW').sum()
    print(f"  Actual RED days in test period   : {n_red}")
    print(f"  Actual YELLOW days in test period: {n_yellow}")

    target_cls = 'RED' if n_red > 0 else 'YELLOW'
    print(f"  Analyzing: {target_cls}")
    print()

    df['is_target'] = df['alert_actual'] == target_cls
    events = []
    cur    = None
    for i, row in df.iterrows():
        if row['is_target']:
            if cur is None: cur = {'idx': i, 'date': row['target_date'], 'days': 1}
            else: cur['days'] += 1
        else:
            if cur: events.append(cur); cur = None
    if cur: events.append(cur)

    print(f"  {target_cls} event blocks: {len(events)}")
    print()

    lead_times = []
    for ev in events:
        eidx      = ev['idx']
        detected  = False
        lead_time = 0
        search_start = max(0, eidx - 14)

        for j in range(eidx - 1, search_start - 1, -1):
            if df.iloc[j]['alert_status'] in ['YELLOW', 'RED']:
                lead_time = eidx - j
                detected  = True
            else:
                break

        if not detected and df.iloc[eidx]['alert_status'] in ['YELLOW', 'RED']:
            detected = True; lead_time = 0

        icon   = "✓" if detected else "✗"
        lt_str = f"{lead_time}d" if detected else "MISSED"
        print(f"    {icon} {ev['date'].date()} ({ev['days']}d {target_cls}) → lead time: {lt_str} | forecast at event: {df.iloc[eidx]['alert_status']}")
        lead_times.append({'detected': detected, 'lead_time': lead_time if detected else None})

    n_det   = sum(1 for lt in lead_times if lt['detected'])
    n_miss  = len(lead_times) - n_det

    print()
    print(f"  Detected : {n_det}/{len(lead_times)}")
    print(f"  Missed   : {n_miss}/{len(lead_times)}")

    result = {'target_class': target_cls, 'n_events': len(events),
              'n_detected': n_det, 'n_missed': n_miss}

    if n_det > 0:
        lt_vals = [lt['lead_time'] for lt in lead_times if lt['detected']]
        print(f"  Mean lead time  : {np.mean(lt_vals):.1f} days  (target >= {TARGETS['Lead_Time_Avg']:.0f})")
        print(f"  Min / Max       : {min(lt_vals)} / {max(lt_vals)} days")
        result['mean_lead_time']   = round(np.mean(lt_vals), 1)
        result['min_lead_time']    = int(min(lt_vals))
        result['max_lead_time']    = int(max(lt_vals))
    else:
        print(f"  Cannot compute lead time — no events detected")
        print(f"  Root cause: same as classification — ARIMA forecasts always GREEN")

    print()
    return result


# =============================================================================
# SUMMARY
# =============================================================================

def generate_summary(accuracy_results, classification_results, lead_time_results):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print()
    print("="*70)
    print(" COMPREHENSIVE RESULTS SUMMARY")
    print("="*70)
    print()

    # Table 4.1
    if accuracy_results:
        print("  TABLE 4.1: FORECAST ACCURACY")
    print("  " + "─"*76)
    print(f"  {'Parameter':<22} {'─── 1-Day Ahead ───':^27}   {'─── 5-Day Ahead ───':^27}")
    print(f"  {'':<22} {'MAE':>6} {'RMSE':>6} {'MAPE':>6} {'MASE':>6}   {'MAE':>6} {'RMSE':>6} {'MAPE':>6} {'MASE':>6}")
    print("  " + "─"*76)
    for param in ['discharge_temp', 'discharge_pressure', 'jacket_water']:
        if param not in accuracy_results: continue
        r    = accuracy_results[param]
        name = param.replace('_', ' ').title()[:21]
        print(f"  {name:<22} "
              f"{r['MAE_1day']:>5.3f} {r['RMSE_1day']:>5.3f} {r['MAPE_1day']:>4.1f}% {r['MASE_1day']:>5.3f}"
              f"   {r['MAE_5day']:>5.3f} {r['RMSE_5day']:>5.3f} {r['MAPE_5day']:>4.1f}% {r['MASE_5day']:>5.3f}")
    print("  " + "─"*76)
    print()

    # Table 4.2
    if classification_results:
        cm = classification_results['class_metrics']
        print("  TABLE 4.2: ALERT CLASSIFICATION")
        print("  " + "─"*55)
        print(f"  {'Class':<10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
        print("  " + "─"*55)
        for cls in ALERT_CLASSES:
            if cls in cm:
                m = cm[cls]
                print(f"  {cls:<10} {m['precision']:>9.1f}% {m['recall']:>9.1f}% {m['f1']:>9.1f}% {m['support']:>10}")
        print("  " + "─"*55)
        print(f"  {'Accuracy':<10} {classification_results['accuracy']:>36.1f}%")
        print()

    # Table 4.3
    if lead_time_results:
        lt = lead_time_results
        print(f"  TABLE 4.3: LEAD TIME ({lt['target_class']} events)")
        print("  " + "─"*40)
        print(f"  {'Events in test period':<25} {lt['n_events']}")
        print(f"  {'Events detected':<25} {lt['n_detected']}")
        print(f"  {'Events missed':<25} {lt['n_missed']}")
        if lt.get('mean_lead_time'):
            print(f"  {'Mean lead time':<25} {lt['mean_lead_time']} days")
        print("  " + "─"*40)
        print()

    # Hypothesis validation
    print("  HYPOTHESIS VALIDATION")
    print("  " + "─"*60)

    if accuracy_results:
        avg_mape = np.mean([accuracy_results[p]['MAPE_1day'] for p in accuracy_results])
        avg_mase = np.mean([accuracy_results[p]['MASE_1day'] for p in accuracy_results])
        h1 = avg_mape < TARGETS['MAPE_1day'] and avg_mase < TARGETS['MASE']
        print(f"  H1 Forecast Accuracy : {'MET' if h1 else 'NOT MET':>10}  (MAPE={avg_mape:.1f}% MASE={avg_mase:.3f})")

    if classification_results:
        acc = classification_results['accuracy']
        h2  = acc > 75
        print(f"  H2 Alert Accuracy    : {'MET' if h2 else 'NOT MET':>10}  (overall={acc:.1f}%)")
        red_rec = classification_results['class_metrics'].get('RED', {}).get('recall', 0)
        print(f"     RED recall        : {red_rec:.1f}%  (0% = all RED missed by forecast)")

    if lead_time_results:
        lt  = lead_time_results.get('mean_lead_time')
        h3  = (lt or 0) >= TARGETS['Lead_Time_Avg']
        lt_str = f"{lt} days" if lt else "N/A — no events detected"
        print(f"  H3 Lead Time         : {'MET' if h3 else 'NOT MET':>10}  ({lt_str})")

    print("  " + "─"*60)
    print()

    # Save JSON
    with open(f'{OUTPUT_DIR}performance_metrics.json', 'w') as f:
        json.dump({'forecast_accuracy': accuracy_results,
                   'classification': classification_results,
                   'lead_time': lead_time_results,
                   'targets': TARGETS}, f, indent=2, default=str)
    print(f"  Results saved: {OUTPUT_DIR}performance_metrics.json")

    # Plots
    if accuracy_results:
        params     = list(accuracy_results.keys())
        labels     = [p.replace('_', '\n').title() for p in params]
        fig, axes  = plt.subplots(1, 4, figsize=(16, 4))
        for ax, metric, title in zip(axes, ['MAE','RMSE','MAPE','MASE'], ['MAE','RMSE','MAPE (%)','MASE']):
            v1 = [accuracy_results[p][f'{metric}_1day'] for p in params]
            v5 = [accuracy_results[p][f'{metric}_5day'] for p in params]
            x  = np.arange(len(params)); w = 0.35
            ax.bar(x - w/2, v1, w, label='1-Day', color='#3B82F6', alpha=0.8)
            ax.bar(x + w/2, v5, w, label='5-Day', color='#EF4444', alpha=0.8)
            ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')
            if metric == 'MASE': ax.axhline(1.0, color='black', linestyle='--', linewidth=1)
        plt.suptitle('Forecast Accuracy by Parameter and Horizon', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}forecast_accuracy.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Plot saved: {OUTPUT_DIR}forecast_accuracy.png")

    if classification_results:
        cm_data = classification_results['confusion_matrix']
        matrix  = np.array([[cm_data.get(fc, {}).get(ac, 0) for ac in ALERT_CLASSES] for fc in ALERT_CLASSES])
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(matrix, cmap='Blues', aspect='auto')
        ax.set_xticks(range(len(ALERT_CLASSES))); ax.set_yticks(range(len(ALERT_CLASSES)))
        ax.set_xticklabels(ALERT_CLASSES); ax.set_yticklabels(ALERT_CLASSES)
        ax.set_xlabel('Actual', fontsize=11); ax.set_ylabel('Forecast', fontsize=11)
        ax.set_title('Alert Classification Confusion Matrix', fontsize=12)
        for i in range(len(ALERT_CLASSES)):
            for j in range(len(ALERT_CLASSES)):
                val   = matrix[i, j]
                color = 'white' if val > matrix.max() * 0.6 else 'black'
                ax.text(j, i, str(val), ha='center', va='center', color=color, fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax, shrink=0.8)
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}confusion_matrix.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Plot saved: {OUTPUT_DIR}confusion_matrix.png")
    print()


# =============================================================================
# MAIN
# =============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print()
    print("="*70)
    print(" VRU COMPRESSOR — CHAPTER 4 PERFORMANCE EVALUATION")
    print(" Section 4.1: Forecast Accuracy (MAE/RMSE/MAPE/MASE)")
    print(" Section 4.2: Alert Classification (Precision/Recall/F1/Lead Time)")
    print("="*70)
    print()

    accuracy       = evaluate_forecast_accuracy()
    classification = evaluate_alert_classification()
    lead_time      = evaluate_lead_time()
    generate_summary(accuracy, classification, lead_time)

    print("="*70)
    print(" EVALUATION COMPLETE")
    print(" Next: open performance_results/ for plots and JSON")
    print("="*70)
    print()


if __name__ == "__main__":
    main()
