"""Пять статистических подходов. На вход поступает только история до origin."""
import warnings
import numpy as np
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt


def fit_forecast(history, steps, candidate):
    values=np.asarray(history,dtype=float)
    if values.ndim != 1 or len(values)<20 or steps<1 or not np.isfinite(values).all() or (values<=0).any():
        raise ValueError('Expected finite positive one-dimensional price history of length >=20')
    scale=float(values[0])
    y=values/scale
    method=candidate['method']
    attempts=1
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        if method=='ar':
            result=AutoReg(y,lags=candidate['lags'],trend='c').fit()
            converged=True
        elif method=='arima':
            model=ARIMA(y,order=tuple(candidate['order']),trend=candidate['trend'],enforce_stationarity=True,enforce_invertibility=True)
            result=model.fit(method_kwargs={'method':'lbfgs','maxiter':200})
            converged=bool(result.mle_retvals.get('converged',False))
            if not converged:
                attempts=2
                result=model.fit(method_kwargs={'method':'powell','maxiter':1000})
                converged=bool(result.mle_retvals.get('converged',False))
        elif method in ['ses','holt','holt_damped']:
            model=SimpleExpSmoothing(y,initialization_method='estimated') if method=='ses' else Holt(y,damped_trend=method=='holt_damped',initialization_method='estimated')
            result=model.fit(optimized=True,use_brute=True,method='L-BFGS-B')
            converged=bool(result.mle_retvals.get('success',False))
        else:
            raise ValueError(f'Unknown statistical method: {method}')
        forecast=np.asarray(result.forecast(steps),dtype=float)*scale
    if not converged:
        raise RuntimeError(f'Optimizer failed after {attempts} attempts')
    if len(forecast)!=steps or not np.isfinite(forecast).all():
        raise ValueError('Invalid forecast')
    # Цена не обрезается принудительно: отрицательный прогноз остаётся видимым дефектом.
    diagnostics={'attempts':attempts,'warnings':[str(w.message) for w in caught],
                 'negative_forecast':bool((forecast<0).any())}
    if method in ['ses','holt','holt_damped']:
        diagnostics['smoothing_parameters']={k:float(result.params[k]) for k in ['smoothing_level','smoothing_trend','damping_trend'] if k in result.params and np.isfinite(result.params[k])}
    return forecast.tolist(), diagnostics

