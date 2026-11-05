"""Нейросетевая валидация с checkpoint и тремя заранее заданными seed."""
import hashlib
import json
import platform
import sys
import time
from importlib.metadata import version
from pathlib import Path
from statistics import mean,stdev
from experiment import ROOT,load_rows,metrics,write_csv
from neural_models import prepare,fit_prepared


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    config_path=ROOT/'config/neural.json'
    grid=json.loads(config_path.read_text(encoding='utf-8'))
    settings=json.loads((ROOT/'config/experiment.json').read_text(encoding='utf-8'))
    source=ROOT/'data/processed/brent_active_provisional.csv'
    if digest(source)!=settings['processed_sha256']:raise ValueError('Data changed')
    out=ROOT/'results/validation';out.mkdir(exist_ok=True,parents=True)
    fingerprint={'config':digest(config_path),'data':digest(source),'split':digest(ROOT/'config/experiment.json'),
        'models_code':digest(ROOT/'src/neural_models.py'),'runner_code':digest(Path(__file__)),
        'torch':version('torch'),'numpy':version('numpy')}
    checkpoint=out/'neural_checkpoint.jsonl';meta=out/'neural_checkpoint_meta.json'
    if meta.exists():
        if json.loads(meta.read_text())!=fingerprint:raise ValueError('Checkpoint fingerprint changed; retain old run and version outputs before rerunning')
    else:meta.write_text(json.dumps(fingerprint,indent=2),encoding='utf-8')
    entries=[]
    if checkpoint.exists():
        entries=[json.loads(line) for line in checkpoint.read_text(encoding='utf-8').splitlines() if line.strip()]
    completed={(r['candidate'],r['seed'],r['horizon'],r['origin_index']) for r in entries}
    if len(completed)!=len(entries):raise ValueError('Duplicate checkpoint keys')
    rows=load_rows()[:settings['test_origin_start']+1]
    prices=[r['close'] for r in rows]
    start=time.perf_counter();cache={}
    for fold in settings['validation_folds']:
        for t in range(fold['origin_start'],fold['origin_end']+1):
            for h in settings['horizons']:
                if t+h>settings['test_origin_start']:raise ValueError('Validation crosses test')
                cache[t,h]=prepare(prices[:t+1],h,grid['lookback'])
    prep_seconds=time.perf_counter()-start
    with checkpoint.open('a',encoding='utf-8') as stream:
        for candidate in grid['candidates']:
            for seed in grid['seeds']:
                for h in settings['horizons']:
                    print(f"Fitting {candidate['id']} seed={seed} h={h}; saved={len(entries)}",flush=True)
                    for fold in settings['validation_folds']:
                        for t in range(fold['origin_start'],fold['origin_end']+1):
                            key=(candidate['id'],seed,h,t)
                            if key in completed:continue
                            entry=dict(candidate=candidate['id'],architecture=candidate['architecture'],seed=seed,horizon=h,fold=fold['id'],
                                origin_index=t,origin_date=rows[t]['timestamp_utc'][:10],target_index=t+h,target_date=rows[t+h]['timestamp_utc'][:10],current=prices[t],actual=prices[t+h])
                            fit_start=time.perf_counter()
                            try:
                                prediction,diagnostics=fit_prepared(cache[t,h],candidate,grid,seed)
                                entry.update(status='ok',predicted=prediction,diagnostics=diagnostics)
                            except Exception as exc:entry.update(status='failed',error=repr(exc))
                            entry['fit_forecast_seconds']=time.perf_counter()-fit_start
                            stream.write(json.dumps(entry)+'\n');stream.flush()
                            entries.append(entry);completed.add(key)
    expected=sum(f['origin_end']-f['origin_start']+1 for f in settings['validation_folds'])
    good=[r for r in entries if r['status']=='ok'];bad=[r for r in entries if r['status']!='ok']
    records=[{k:v for k,v in r.items() if k not in ['diagnostics','status']} for r in good]
    write_csv(out/'neural_predictions.csv',records)
    summaries=[]
    for c in grid['candidates']:
        for h in settings['horizons']:
            for seed in grid['seeds']:
                pool=[r for r in good if r['candidate']==c['id'] and r['horizon']==h and r['seed']==seed]
                if len(pool)!=expected:continue
                for fold in ['all']+[f['id'] for f in settings['validation_folds']]:
                    sub=[r for r in pool if fold=='all' or r['fold']==fold]
                    score=metrics([r['actual'] for r in sub],[r['predicted'] for r in sub],[r['current'] for r in sub])
                    summaries.append(dict(candidate=c['id'],architecture=c['architecture'],horizon=h,seed=seed,fold=fold,**score))
    write_csv(out/'neural_seed_metrics.csv',summaries)
    aggregate=[]
    for c in grid['candidates']:
        for h in settings['horizons']:
            for fold in ['all']+[f['id'] for f in settings['validation_folds']]:
                sub=[r for r in summaries if r['candidate']==c['id'] and r['horizon']==h and r['fold']==fold]
                if len(sub)!=len(grid['seeds']):continue
                row=dict(candidate=c['id'],architecture=c['architecture'],horizon=h,fold=fold,seeds=len(sub),n_origins=sub[0]['n'])
                for m in ['MAE','RMSE','sMAPE_pct','direction_accuracy_pct','direction_coverage_pct']:
                    row[m+'_mean']=mean(r[m] for r in sub);row[m+'_seed_sd']=stdev(r[m] for r in sub)
                aggregate.append(row)
    write_csv(out/'neural_metrics.csv',aggregate)
    winners={str(h):min([r for r in aggregate if r['horizon']==h and r['fold']=='all'],key=lambda r:(r['RMSE_mean'],r['MAE_mean'],r['candidate']))['candidate'] for h in settings['horizons']}
    report=dict(scope='validation_only',winners=winners,selection=grid['selection'],successful_fits=len(good),failed_fits=len(bad),seeds=grid['seeds'],
        expected_origins=expected,total_fit_seconds=sum(r['fit_forecast_seconds'] for r in entries),current_invocation_seconds=time.perf_counter()-start,
        current_feature_seconds=prep_seconds,test_evaluated=False,fingerprint=fingerprint,python=sys.version,platform=platform.platform())
    (out/'neural_run.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)
    for r in aggregate:
        if r['fold']=='all':print(r['candidate'],r['horizon'],'RMSE',round(r['RMSE_mean'],4),'seed_sd',round(r['RMSE_seed_sd'],4),flush=True)

if __name__=='__main__':
    from pathlib import Path
    main()

