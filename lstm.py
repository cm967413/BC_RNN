import torch
import torch.nn as nn
from torch.autograd import Variable

class lstm(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, unit_num, device):
        super(lstm, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.unit_num = unit_num
        self.device = device
        
        self.embed = nn.Linear(self.input_dim, self.hidden_dim)
        self.lstm = nn.ModuleList([nn.LSTMCell(self.hidden_dim, self.hidden_dim) for _ in range(self.unit_num)])
        self.output = nn.Linear(self.hidden_dim, self.output_dim)
        self.hidden = []

    def init_hidden(self, batch_size):
        hidden = []
        for _ in range(self.unit_num):
            hidden.append((Variable(torch.zeros(batch_size, self.hidden_dim).to(self.device)),
                           Variable(torch.zeros(batch_size, self.hidden_dim).to(self.device))))
        return hidden

    def forward(self, x_in):
        embedded = self.embed(x_in)
        h_in = embedded
        for i in range(self.unit_num):
            self.hidden[i] = self.lstm[i](h_in, self.hidden[i])
            h_in = self.hidden[i][0]
            
        return self.output(h_in)