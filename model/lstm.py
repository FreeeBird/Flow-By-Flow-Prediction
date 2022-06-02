import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layer=3, seq_len=12, pre_len=3):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            in_dim, hidden_dim, n_layer, batch_first=True, dropout=0.5)
        print(self.lstm)
        self.fc = nn.Linear(hidden_dim, 1)
        self.time_linear = nn.Linear(seq_len, pre_len)

    def forward(self, x):
        # BS,seq_len = x.size()
        x = x.unsqueeze(-1)  # bs,t,1
        x, _ = self.lstm(x)  # BS,T,h
        x = self.fc(x)  # BS,T,1
        x = self.time_linear(x.squeeze(-1))

        return x

class LSTM_TM(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layer=3, seq_len=12, pre_len=3):
        super(LSTM_TM, self).__init__()
        self.lstm = nn.LSTM(
            in_dim, hidden_dim, n_layer, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(hidden_dim, in_dim)
        self.time_linear = nn.Linear(seq_len, pre_len)


    def forward(self, x):
        # BS,seq_len,f = x.size()
        x, _ = self.lstm(x)  # BS,T,h
        x = self.fc(x)  # BS,T,f
        return x[:,-1]


