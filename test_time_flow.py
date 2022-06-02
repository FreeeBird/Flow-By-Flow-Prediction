from importlib.resources import path
import time
import os
import numpy as np
import torch.nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error
import argparse
import sys, os
from tools.data_process import *
from tools.tool import *

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='lstm', help='train model name')
parser.add_argument('--dataset', default='nobel', help='chose dataset', choices=['geant', 'abilene','nobel','germany'])
parser.add_argument('--gpu', default=0, help='use -1/0/1 chose cpu/gpu:0/gpu:1', choices=[-1, 0, 1])
parser.add_argument('--seq_len', default=12, help='input history length')
parser.add_argument('--pre_len', default=1, help='prediction length')
parser.add_argument('--dim_model', default=32, help='dimension of embedding vector')
parser.add_argument('--dim_attn', default=32, help='dimension of attention')
parser.add_argument('--num_heads', default=1, help='attention  heads')
parser.add_argument('--train_rate', default=0.7, help='')
parser.add_argument('--rnn_layers', default=3, help='rnn layers')
parser.add_argument('--encoder_layers', default=3, help='encoder layers')
parser.add_argument('--dropout', default=0.5, help='dropout rate')

args = parser.parse_args()
if args.gpu == 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,0"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# param
dataset = args.dataset
fea_path = get_data_path(dataset)
num_nodes = get_data_nodes(dataset)
num_flows = num_nodes * num_nodes

def get_model_path(model,dataset='nobel'):
    paths = [
        [
            'LSTM_abilene_12-1_12-22_14:24:50_dict.pkl','LSTM_geant_12-1_01-18_12:25:17_dict.pkl','LSTM_germany_12-1_01-16_15:12:44_dict.pkl','LSTM_nobel_12-1_01-12_21:04:47_dict.pkl'
        ],[
            'GRU_abilene_12-1_12-31_11:18:52_dict.pkl','GRU_geant_12-1_01-18_13:00:44_dict.pkl','GRU_germany_12-1_01-21_10:59:46_dict.pkl','GRU_nobel_12-1_01-12_21:08:56_dict.pkl'
        ],[
            'Transformer_abilene_12-1_12-30_21:26:48_dict.pkl','Transformer_geant_12-1_01-09_20:52:40_dict.pkl','Transformer_germany_12-1_01-16_23:24:14_dict.pkl','Transformer_nobel_12-1_01-12_21:36:19_dict.pkl'
        ]
    ]
    model_dick = {'lstm':0,'gru':1,'transformer':2}
    data_dict = {'abilene':0,'geant':1,'germany':2,'nobel':3}
    return '/home/liyiyong/flow-wise-prediction/dict/'+ paths[model_dick[model]][data_dict[dataset]]

model_path = get_model_path(args.model,args.dataset)   

seq_len = args.seq_len
pre_len = args.pre_len
em_size = args.dim_model
num_head = args.num_heads
train_rate = args.train_rate

################# data
# load data
data = np.load(fea_path)
# split dataset
_, _, _, _, test_x, test_y, max_data = split_dataset(data, train_rate=train_rate,val_rate=0.1,
                                                                                seq_len=seq_len,
                                                                                predict_len=pre_len)
# ndarray -> tensor
numer_of_matrix = test_y.shape[0]
rep = int(10000/numer_of_matrix)+1
test_x, test_y, = torch.from_numpy(test_x).float(), torch.from_numpy(test_y).float()
print(test_x.shape,test_y.shape)
test_x = test_x.repeat((rep,1,1))[:10000]
test_y = test_y.repeat((rep,1,1))[:10000]
numer_of_matrix = test_y.shape[0]
# tensor dataset

test_dataset = TensorDataset(test_x, test_y)
# dataloader
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, pin_memory=True, num_workers=8)
# print(len(test_loader))
################# model
# torch.cuda.synchronize()

model = get_model(args.model, args=args)
model.load_state_dict(torch.load(model_path))
model = model.to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f'{total_params:,} total parameters.')
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f'{total_trainable_params:,} training parameters.')
print(args)
model.eval()
# test_y = test_y*max_data
########## test ###########
# torch.cuda.synchronize()
test_times = []
for i in range(5):
    # test_start = time.time()
    y_true, y_pred = [], []
    pre_time = []
    spand_times = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            x = x.reshape([-1,x.shape[-1]])
            torch.cuda.synchronize()
            temp_time = time.time()
            y_hat = model(x)
            torch.cuda.synchronize()
            temp_time = time.time() - temp_time
            spand_times += temp_time
            # print(temp_time)
            y = y.squeeze(-1)
            y_hat = y_hat.reshape([-1,num_flows])
            y = y.cpu().detach().numpy()
            y_hat = y_hat.cpu().detach().numpy()
            # numer_of_matrix += y.shape[0]
            # print(y.shape[0])
            # pre_time.append(temp_time / y.shape[0])
            y_pred.extend(y_hat)
            y_true.extend(y)
    # print(prof.key_averages().table(sort_by="self_cpu_time_total"))
    # torch.cuda.synchronize()

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    print('TEST RESULT:',
        'mse:{:.6}'.format(mse),
        'mae:{:.6}'.format(mae),
        )
    test_times.append(spand_times)

print("test time: ", test_times)
print("test time: ", np.mean(test_times))
print("numer_of_matrix: ",  numer_of_matrix)
print("test time per matrix: ", np.mean(test_times) / numer_of_matrix)


