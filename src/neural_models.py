"""Небольшие нейросети с одинаковой входной историей и обучением без будущего."""
import random
import numpy as np
import torch
from torch import nn


def configure(seed,threads=1):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def prepare(history,horizon,lookback=20):
    p=np.asarray(history,dtype=float)
    if p.ndim!=1 or len(p)<lookback+horizon+2 or horizon not in [1,5] or not np.isfinite(p).all() or (p<=0).any():
        raise ValueError('Invalid neural history/horizon')
    returns=p[1:]/p[:-1]-1
    origins=np.arange(lookback,len(p)-horizon)
    # return[k] относится к переходу price[k] -> price[k+1].
    X=np.stack([returns[j-lookback:j] for j in origins])[:,:,None]
    y=np.array([p[j+horizon]-p[j] for j in origins])
    current=returns[-lookback:][None,:,None]
    xm=float(X.mean());xs=float(X.std());ym=float(y.mean());ys=float(y.std())
    xs=xs if xs>1e-12 else 1.;ys=ys if ys>1e-12 else 1.
    return {'X':torch.tensor((X-xm)/xs,dtype=torch.float32),
        'y':torch.tensor((y-ym)/ys,dtype=torch.float32),
        'current':torch.tensor((current-xm)/xs,dtype=torch.float32),
        'price':float(p[-1]),'x_mean':xm,'x_scale':xs,'y_mean':ym,'y_scale':ys,
        'training_n':len(y),'last_training_origin':int(origins[-1]),'last_training_target':int(origins[-1]+horizon)}


class ForecastNet(nn.Module):
    def __init__(self,architecture,hidden_size=8,lookback=20):
        super().__init__();self.architecture=architecture
        if architecture=='mlp':
            self.network=nn.Sequential(nn.Flatten(),nn.Linear(lookback,hidden_size),nn.Tanh(),nn.Linear(hidden_size,1))
        elif architecture in ['rnn','lstm','gru']:
            cls={'rnn':nn.RNN,'lstm':nn.LSTM,'gru':nn.GRU}[architecture]
            self.recurrent=cls(1,hidden_size,num_layers=1,batch_first=True,bidirectional=False)
            self.head=nn.Linear(hidden_size,1)
        elif architecture=='cnn':
            self.network=nn.Sequential(nn.Conv1d(1,hidden_size,kernel_size=3,padding=0),nn.Tanh(),nn.Flatten(),nn.Linear(hidden_size*(lookback-2),1))
        else:raise ValueError('Unknown architecture')
    def forward(self,x):
        if self.architecture in ['rnn','lstm','gru']:
            outputs,_=self.recurrent(x)
            return self.head(outputs[:,-1,:]).squeeze(-1)
        if self.architecture=='cnn':x=x.transpose(1,2)
        return self.network(x).squeeze(-1)


def fit_prepared(data,candidate,settings,seed):
    configure(seed,settings['threads'])
    model=ForecastNet(candidate['architecture'],candidate['hidden_size'],settings['lookback'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=settings['learning_rate'],weight_decay=settings['weight_decay'])
    loss_fn=nn.MSELoss()
    model.train()
    with torch.no_grad():initial=float(loss_fn(model(data['X']),data['y']).item())
    for _ in range(settings['epochs']):
        optimizer.zero_grad(set_to_none=True)
        loss=loss_fn(model(data['X']),data['y'])
        if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),settings['gradient_clip'],error_if_nonfinite=True)
        optimizer.step()
    model.eval()
    with torch.no_grad():
        final=float(loss_fn(model(data['X']),data['y']).item())
        delta=float(model(data['current']).item())*data['y_scale']+data['y_mean']
    predicted=data['price']+delta
    if not np.isfinite(predicted) or not np.isfinite(final):raise ValueError('Nonfinite forecast/loss')
    diagnostics={k:data[k] for k in ['training_n','last_training_origin','last_training_target','x_mean','x_scale','y_mean','y_scale']}
    diagnostics.update(parameter_count=sum(p.numel() for p in model.parameters()),epochs=settings['epochs'],initial_train_mse_standardized=initial,
        final_train_mse_standardized=final,negative_forecast=predicted<0)
    return predicted,diagnostics
