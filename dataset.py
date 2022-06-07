import math
import torch
from torch.utils.data import Dataset, DataLoader

class RNN_Dataset(Dataset):
    def __init__(self, X, Y, L, device):
        self.X = X
        self.Y = Y
        self.L = L
        self.device = device
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        x = self.X[index].to(self.device)
        y = self.Y[index].to(self.device)
        l = self.L[index]
        return x, y, l

def get_dataloader(X, Y, L, batch_size=32, ratio=0.8, device='cpu'):
    train_x, val_x = X[:math.ceil(len(X)*ratio)], X[math.ceil(len(Y)*ratio):]
    train_y, val_y = Y[:math.ceil(len(X)*ratio)], Y[math.ceil(len(Y)*ratio):]
    train_l, val_l = L[:math.ceil(len(X)*ratio)], L[math.ceil(len(Y)*ratio):]

    train_set = RNN_Dataset(train_x, train_y, train_l, device=device)
    val_set = RNN_Dataset(val_x, val_y, val_l, device=device)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    print(f'training size: {sum(train_l)}, validation size: {sum(val_l)}')

    return train_loader, val_loader