import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from tqdm.notebook import tqdm
from layers import SinkhornDistance

def get_data(root):
    X, Y = list(), list()
    progress = tqdm(total = len(os.listdir(root)))
    for file in os.listdir(root):
        df = pd.read_excel(f'{root}/{file}')
        X.append(np.array(df.drop(['time', 'propofol'], axis=1)).astype(np.float32))
        Y.append(np.array(df['propofol']).astype(np.float32))
        progress.update(1)

    X = [torch.tensor(x) for x in X]
    Y = [torch.tensor(y) for y in Y]
    L = [len(x) for x in X]
    
    X = rnn_utils.pad_sequence(X, batch_first=True)
    Y = rnn_utils.pad_sequence(Y, batch_first=True)
    
    return X, Y, L

def get_state_diff_data(root):
  X, Y = list(), list()
  progress = tqdm(total = len(os.listdir(root)))
  for file in os.listdir(root):
    df = pd.read_excel(f'{root}/{file}')
    df = df.drop(['time'], axis=1)
    column_name = df.columns
    for col in column_name:
      if col == 'propofol':
        continue
      tmp = list()
      tmp.append(0)
      for i in range(1,len(df[col])):
        tmp.append(df[col][i] - df[col][i-1])
      new_col = col + '_diff'
      df[new_col] = tmp
      
    X.append(np.array(df.drop(['propofol'], axis=1)).astype(np.float32))
    Y.append(np.array(df['propofol']).astype(np.float32))
    progress.update(1)

  X = [torch.tensor(x) for x in X]
  Y = [torch.tensor(y) for y in Y]
  L = [len(x) for x in X]

  X = rnn_utils.pad_sequence(X, batch_first=True)
  Y = rnn_utils.pad_sequence(Y, batch_first=True)

  return X, Y, L

def mse_loss(y_true, y_pred, l, gtn):
    mse_criterion = nn.MSELoss()
    return mse_criterion(y_true, y_pred) * (l - gtn)

def wassdistance_loss(y_true, y_pred, l, gtn):
    sinkhorn = SinkhornDistance(eps=0.1, max_iter=1000, reduction=None)
    
    x = torch.tensor([[i*1.0 / (l - gtn)] for i in range(l - gtn)])
    y_true = torch.cat([x, y_true.reshape(-1, 1).cpu()], 1)
    y_pred = torch.cat([x, y_pred.reshape(-1, 1).cpu()], 1)
    dist, P, C = sinkhorn(y_true, y_pred)
    return dist.cuda() * (l - gtn)

def mixture_loss(y_true, y_pred, l, gtn):
    loss_1 = mse_loss(y_true, y_pred, l, gtn)
    loss_2 = wassdistance_loss(y_true, y_pred, l, gtn)
    return loss_1 + loss_2

class teacher_forcing_rate():
    def __init__(self, tfr_decay_rate=0.05, tfr_lower_bound=0.2):
        self.tfr = 1
        self.tfr_decay_rate = tfr_decay_rate
        self.tfr_lower_bound = tfr_lower_bound
        
    def update(self):
        self.tfr = max(self.tfr - self.tfr_decay_rate, self.tfr_lower_bound)
    
    def get_tfr(self):
        return self.tfr