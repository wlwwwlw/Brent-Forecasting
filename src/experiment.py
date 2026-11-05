"""Общие правила эксперимента: временные границы, прогнозы и метрики."""
import csv
import math
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[1]

def load_rows():
    with (ROOT/'data/processed/brent_active_provisional.csv').open(encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in ['open', 'high', 'low', 'close', 'volume']:
            row[key] = float(row[key])
    return rows

def sign(value):
    return (value > 0) - (value < 0)

def metrics(actual, predicted, current):
    """sMAPE в процентах; направление - знак изменения относительно текущей цены."""
    if not actual or not len(actual) == len(predicted) == len(current):
        raise ValueError('Metric inputs must be nonempty and equally sized')
    if not all(math.isfinite(v) for seq in [actual, predicted, current] for v in seq):
        raise ValueError('Nonfinite metric input')
    errors = [p-y for y,p in zip(actual,predicted)]
    smape = [0.0 if abs(y)+abs(p) == 0 else 200*abs(p-y)/(abs(y)+abs(p)) for y,p in zip(actual,predicted)]
    directions = [sign(p-c) for p,c in zip(predicted,current)]
    true_directions = [sign(y-c) for y,c in zip(actual,current)]
    return dict(n=len(actual), MAE=mean(abs(e) for e in errors),
                RMSE=math.sqrt(mean(e*e for e in errors)), sMAPE_pct=mean(smape),
                direction_accuracy_pct=100*mean(p == y for p,y in zip(directions,true_directions)),
                direction_coverage_pct=100*mean(p != 0 for p in directions))

def eligible_training_origins(origin, horizon, lookback):
    """В момент origin известны только цели j+h <= origin."""
    if horizon < 1 or lookback < 1:
        raise ValueError('Positive horizon and lookback required')
    return list(range(lookback-1, origin-horizon+1))

def forecast_simple(history, horizon, model, window=None):
    """Получает только доступную историю, будущий ряд в функцию не передаётся."""
    if not history or horizon < 1:
        raise ValueError('History and positive horizon required')
    if model == 'naive':
        return history[-1]
    if model == 'drift':
        if len(history) < 2:
            raise ValueError('Drift needs two observations')
        return history[-1] + horizon*(history[-1]-history[0])/(len(history)-1)
    if model in ['rolling_mean','rolling_median']:
        if not window or len(history) < window:
            raise ValueError('Insufficient window')
        return (mean if model == 'rolling_mean' else median)(history[-window:])
    if model == 'seasonal_naive':
        if not window or len(history) < window:
            raise ValueError('Insufficient seasonal history')
        # Повтор последних m наблюдений: h=1 берёт y[t-m+1], h=m берёт y[t].
        return history[-window + (horizon-1) % window]
    raise ValueError(f'Unknown model {model}')

def write_csv(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:
        raise ValueError('No rows to write')
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
