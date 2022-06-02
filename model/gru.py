import torch
import torch.nn as nn


class GRU(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layer=3, seq_len=12, pre_len=1, dropout=0.3):
        super(GRU, self).__init__()
        self.rnn = nn.GRU(input_size=in_dim, hidden_size=hidden_dim, num_layers=n_layer, batch_first=True,
                          dropout=dropout)
        self.fc = nn.Linear(hidden_dim, in_dim)
        self.time_linear = nn.Linear(seq_len, pre_len)

    def forward(self, x):
        # BS,seq_len = x.size()
        x = x.unsqueeze(-1)  # bs,t,1
        x, _ = self.rnn(x)  # BS,T,h
        x = self.fc(x)  # BS,T,1
        x = self.time_linear(x.squeeze(-1))
        return x

class GRU_TM(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layer=3, seq_len=12, pre_len=1, dropout=0.3):
        super(GRU_TM, self).__init__()
        self.rnn = nn.GRU(input_size=in_dim, hidden_size=hidden_dim, num_layers=n_layer, batch_first=True,
                          dropout=dropout)
        self.fc = nn.Linear(hidden_dim, in_dim)
        self.time_linear = nn.Linear(seq_len, pre_len)

    def forward(self, x):
        # BS,seq_len = x.size()
        x, _ = self.rnn(x)  # BS,T,h
        x = self.fc(x)  # BS,T,f
        # x = self.time_linear(x.permute(0,2,1)).permute(0,2,1).squeeze(1)
        return x[:,-1]


# model = GRU(1,32,3,12,1)
# print(model)
# model = GRU_TM(144,144,3,12,1)
# print(model)