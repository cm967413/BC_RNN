import random
import numpy as np
import torch

def train(model, optimizer, x, y, l, tfr, gtn, threshold, loss_func):
    model.zero_grad()
    model.hidden = model.init_hidden(x.shape[0])
    y = y.reshape(y.shape[0], y.shape[1], 1)
    
    loss = 0
    count = 0
    acc_count = 0
    y_pred = y[:, 0]
    pred_list = [y_pred]
    
    use_teacher_forcing = True if random.random() < tfr.get_tfr() else False
    
    for i in range(1, x.shape[1]):
        if use_teacher_forcing or i < gtn:
            y_pred = model(torch.cat([x[:, i], y[:, i-1]], 1)).reshape(-1, 1)
        else:
            y_pred = model(torch.cat([x[:, i], y_pred], 1)).reshape(-1, 1)
        pred_list.append(y_pred)
    
    result = torch.cat(pred_list, 1).reshape(y.shape[0], y.shape[1], 1)
    for i, pred in enumerate(result):
        
        if l[i] <= gtn:
            continue
            
        y_true = y[i, gtn:l[i]]
        y_pred = result[i, gtn:l[i]]
        loss += loss_func(y_true, y_pred, l[i], gtn)
        acc_count += torch.count_nonzero(torch.abs((y_pred - y_true)) <= threshold)
        count += (l[i] - gtn)
        
    loss.backward()
    optimizer.step()
    
    return loss.item(), acc_count, count

def pred(model, x, y, l, gtn, threshold, loss_func):
    model.hidden = model.init_hidden(x.shape[0])
    y = y.reshape(y.shape[0], y.shape[1], 1)
    
    loss = 0
    count = 0
    acc_count = 0
    y_pred = y[:, 0]
    pred_list = [y_pred]
    
    for i in range(1, x.shape[1]):
        if i < gtn:
            y_pred = model(torch.cat([x[:, i], y[:, i-1]], 1)).reshape(-1, 1)
        else:
            y_pred = model(torch.cat([x[:, i], y_pred], 1)).reshape(-1, 1)
        pred_list.append(y_pred)
    
    result = torch.cat(pred_list, 1).reshape(y.shape[0], y.shape[1], 1)
    for i, pred in enumerate(result):
        
        if l[i] <= gtn:
            continue
            
        y_true = y[i, gtn:l[i]]
        y_pred = result[i, gtn:l[i]]
        loss += loss_func(y_true, y_pred, l[i], gtn)
        acc_count += torch.count_nonzero(torch.abs((y_pred - y_true)) <= threshold)
        count += (l[i] - gtn)
    
    return loss.item(), acc_count, count