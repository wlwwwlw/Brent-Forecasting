"""Причинные признаки и прямой прогноз изменения цены на h сессий."""
import warnings
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge,ElasticNet
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.exceptions import ConvergenceWarning

LOOKBACK=20
FEATURE_NAMES=([f'return_lag_{k}' for k in [1,2,3,5,10]]+
               [f'momentum_{w}' for w in [5,10,20]]+
               [f'volatility_{w}' for w in [5,10,20]]+
               [f'price_to_mean_{w}' for w in [5,10,20]])

def features(history):
    """Последнее значение history уже известно; нужны 21 цена для 20 изменений."""
    p=np.asarray(history,dtype=float)
    if p.ndim!=1 or len(p)<LOOKBACK+1 or not np.isfinite(p).all() or (p<=0).any():
        raise ValueError('Features require >=21 finite positive prices')
    returns=p[1:]/p[:-1]-1
    values=([returns[-k] for k in [1,2,3,5,10]]+
            [p[-1]/p[-w-1]-1 for w in [5,10,20]]+
            [np.std(returns[-w:],ddof=1) for w in [5,10,20]]+
            [p[-1]/np.mean(p[-w:])-1 for w in [5,10,20]])
    return np.asarray(values,dtype=float)

def prepare(history,horizon):
    p=np.asarray(history,dtype=float)
    if horizon not in [1,5] or len(p)<LOOKBACK+horizon+2:
        raise ValueError('Insufficient history or unsupported horizon')
    origins=np.arange(LOOKBACK,len(p)-horizon)
    X=np.stack([features(p[:j+1]) for j in origins])
    y=np.array([p[j+horizon]-p[j] for j in origins])
    current=features(p).reshape(1,-1)
    return {'X':X,'y':y,'current':current,'price':float(p[-1]),'origins':origins,'last_target':int(origins[-1]+horizon)}

def estimator(candidate,seed=42):
    m=candidate['method']
    if m=='ridge':regressor=Ridge(alpha=candidate['alpha'])
    elif m=='elastic':regressor=ElasticNet(alpha=candidate['alpha'],l1_ratio=candidate['l1_ratio'],max_iter=20000,tol=1e-6,selection='cyclic')
    elif m=='forest':regressor=RandomForestRegressor(n_estimators=candidate['n_estimators'],max_depth=candidate['max_depth'],min_samples_leaf=candidate['min_samples_leaf'],max_features=1.0,bootstrap=True,random_state=seed,n_jobs=1)
    elif m=='boost':regressor=GradientBoostingRegressor(n_estimators=candidate['n_estimators'],max_depth=candidate['max_depth'],min_samples_leaf=candidate['min_samples_leaf'],learning_rate=candidate['learning_rate'],loss='squared_error',subsample=1.0,random_state=seed,n_iter_no_change=None)
    elif m=='svr':regressor=SVR(C=candidate['C'],epsilon=candidate['epsilon'],kernel='rbf',gamma='scale')
    else:raise ValueError(f'Unknown method {m}')
    return TransformedTargetRegressor(regressor=Pipeline([('scale',StandardScaler()),('model',regressor)]),transformer=StandardScaler())

def fit_prepared(data,candidate,seed=42):
    model=estimator(candidate,seed)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model.fit(data['X'],data['y'])
        delta=float(model.predict(data['current'])[0])
    if any(issubclass(w.category,ConvergenceWarning) for w in caught):
        raise RuntimeError('Estimator failed to converge')
    predicted=data['price']+delta
    if not np.isfinite(predicted):raise ValueError('Nonfinite prediction')
    diagnostics={'training_n':len(data['y']),'last_training_origin':int(data['origins'][-1]),'last_training_target':data['last_target'],
        'scaler_n':int(model.regressor_.named_steps['scale'].n_samples_seen_),
        'target_mean':float(model.transformer_.mean_[0]),'target_scale':float(model.transformer_.scale_[0]),
        'warnings':[str(w.message) for w in caught],'negative_forecast':bool(predicted<0)}
    return predicted,diagnostics

def fit_predict(history,horizon,candidate,seed=42):
    return fit_prepared(prepare(history,horizon),candidate,seed)
