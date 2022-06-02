'''
Author: FreeeBird
Date: 2022-03-10 16:17:24
LastEditTime: 2022-05-25 11:56:11
LastEditors: FreeeBird
Description: 
FilePath: /flow-wise-prediction/model/LSTNet.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTNet(nn.Module):
    def __init__(self, flows=144,seq_len=12,pre_len=1,hidCNN=144,hidRNN=144,hidSkip=10,CNN_kernel=6,skip=2):
        super(LSTNet, self).__init__()
        self.P = seq_len
        self.m = flows
        self.hidR = hidRNN
        self.hidC = hidCNN
        self.hidS = hidSkip
        self.Ck = CNN_kernel
        self.skip = skip
        self.pt = (self.P - self.Ck)//self.skip
        self.hw = seq_len
        self.conv1 = nn.Conv2d(1, self.hidC, kernel_size = (self.Ck, self.m))
        self.GRU1 = nn.GRU(self.hidC, self.hidR)
        self.dropout = nn.Dropout(0.5)
        if (self.skip > 0):
            self.GRUskip = nn.GRU(self.hidC, self.hidS)
            self.linear1 = nn.Linear(self.hidR + self.skip * self.hidS, self.m)
        else:
            self.linear1 = nn.Linear(self.hidR, self.m)
        if (self.hw > 0):
            self.highway = nn.Linear(self.hw, 1)
        # self.output = None
        # if (args.output_fun == 'sigmoid'):
        self.output = nn.Sigmoid()
        # if (args.output_fun == 'tanh'):
            # self.output = F.tanh;
 
    def forward(self, x):
        batch_size = x.size(0)
        
        #CNN
        c = x.view(-1, 1, self.P, self.m)
        c = F.relu(self.conv1(c))
        c = self.dropout(c)
        c = torch.squeeze(c, 3)
        
        # RNN 
        r = c.permute(2, 0, 1).contiguous()
        _, r = self.GRU1(r)
        r = self.dropout(torch.squeeze(r,0))

        
        #skip-rnn
        
        if (self.skip > 0):
            s = c[:,:, int(-self.pt * self.skip):].contiguous()
            s = s.view(batch_size, self.hidC, self.pt, self.skip)
            s = s.permute(2,0,3,1).contiguous()
            s = s.view(self.pt, batch_size * self.skip, self.hidC)
            _, s = self.GRUskip(s)
            s = s.view(batch_size, self.skip * self.hidS)
            s = self.dropout(s)
            r = torch.cat((r,s),1)
        
        res = self.linear1(r)
        
        if (self.hw > 0):
            z = x[:, -self.hw:, :]
            z = z.permute(0,2,1).contiguous().view(-1, self.hw)
            z = self.highway(z)
            z = z.view(-1,self.m)
            res = res + z

        if (self.output):
            res = self.output(res)
        return res
    
        
        
        
# model = LSTNet(529,8,2,529,529,10,6,2)
# x = torch.zeros([32,8,529])
# x = model(x)
# print(x.size())