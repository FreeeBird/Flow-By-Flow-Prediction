import time
import os
from tkinter.tix import Tree
import numpy as np
import torch.nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error
import argparse

from tools.data_process import *
from tools.tool import *



parser = argparse.ArgumentParser()
parser.add_argument('--model', default='lstm_tm', help='train model name')
parser.add_argument('--dataset', default='nobel', help='chose dataset', choices=['geant', 'abilene','nobel','germany'])
parser.add_argument('--gpu', default=0, help='use -1/0/1 chose cpu/gpu:0/gpu:1', choices=[-1, 0, 1])
parser.add_argument('--seq_len', default=12, help='input history length')
parser.add_argument('--pre_len', default=1, help='prediction length')
parser.add_argument('--num_flows', default=17*17, help='dimension of embedding vector')
parser.add_argument('--dim_model', default=17*17, help='dimension of embedding vector')
parser.add_argument('--dim_attn', default=32, help='dimension of attention')
parser.add_argument('--num_heads', default=1, help='attention heads')
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
args.num_flows = num_nodes * num_nodes
num_flows = num_nodes * num_nodes
args.dim_model = args.num_flows
def get_model_path(model,dataset='nobel'):
    paths = [
        [
            'LSTM_TM_abilene_12-1_12-31_16:40:34_dict.pkl','LSTM_TM_geant_12-1_12-31_19:50:52_dict.pkl','LSTM_TM_germany_12-1_01-22_18:04:49_dict.pkl','LSTM_TM_nobel_12-1_01-12_22:01:48_dict.pkl'
        ],[
            'GRU_TM_abilene_12-1_12-31_23:03:56_dict.pkl','GRU_TM_geant_12-1_12-31_21:27:37_dict.pkl','GRU_TM_germany_12-1_01-22_20:16:44_dict.pkl','GRU_TM_nobel_12-1_01-12_22:26:20_dict.pkl'
        ],[
            'Transformer_TM_abilene_12-1_01-01_20:33:44_dict.pkl','Transformer_TM_geant_12-1_01-01_22:36:05_dict.pkl','Transformer_TM_germany_12-1_01-22_20:57:51_dict.pkl','Transformer_TM_nobel_12-1_01-13_09:16:00_dict.pkl'
        ]
    ]
    model_dick = {'lstm_tm':0,'gru_tm':1,'transformer_tm':2}
    data_dict = {'abilene':0,'geant':1,'germany':2,'nobel':3}
    return '/home/liyiyong/flow-wise-prediction/dict/'+ paths[model_dick[model]][data_dict[dataset]]



model_path = get_model_path(args.model,args.dataset)   

seq_len = args.seq_len
pre_len = args.pre_len
em_size = args.dim_model
num_head = args.num_heads
train_rate = args.train_rate
args.dim_model

################# data
# load data
data = np.load(fea_path)
# split dataset
_, _, _, _, test_x, test_y, max_data = split_dataset(data, train_rate=train_rate,val_rate=0.1,
                                                                                seq_len=seq_len,
                                                                                predict_len=pre_len,wise='matrix')
# ndarray -> tensor
numer_of_matrix = test_y.shape[0]
rep = int(10000/numer_of_matrix)+1
test_x, test_y, = torch.from_numpy(test_x).float(), torch.from_numpy(test_y).float()
test_x = test_x.repeat((rep,1,1))[:10000]
test_y = test_y.repeat((rep,1))[:10000]
numer_of_matrix = test_y.shape[0]
# tensor dataset

test_dataset = TensorDataset(test_x, test_y)
# dataloader
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, pin_memory=False, num_workers=8)

################# model
# torch.cuda.synchronize()

model = get_model(args.model, args=args)
model.load_state_dict(torch.load(model_path))
model = model.to(device)
# torch.cuda.synchronize()
# print(time.time()- test_start, ' s')
total_params = sum(p.numel() for p in model.parameters())
print(f'{total_params:,} total parameters.')
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f'{total_trainable_params:,} training parameters.')
print(args)

# test_y = test_y*max_data
########## test ###########
# torch.cuda.synchronize()
model.eval()
test_times = []
for i in range(5):
    test_start = time.time()
    y_true, y_pred = [], []
    pre_time = []
    spand_times = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            temp_time = time.time()
            y_hat = model(x)
            torch.cuda.synchronize()
            temp_time = time.time() - temp_time
            spand_times += temp_time
            y = np.reshape(y.cpu().detach().numpy(), [-1, num_flows])
            y_hat = np.reshape(y_hat.cpu().detach().numpy(), [-1, num_flows])
            # numer_of_matrix += y.shape[0]
            y_pred.extend(y_hat)
            y_true.extend(y)
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)


        print('TEST RESULT:',
            'mse:{:.6}'.format(mse),
            'mae:{:.6}'.format(mae),
            )
    # torch.cuda.synchronize()
    # test_end = time.time()
    # test_times.append(test_end - test_start)
    test_times.append(spand_times)
    # print("test time: ", (test_end - test_start), ' S')
    # print("numer_of_matrix: ", numer_of_matrix)
    # print("predict time for single matrix:",(test_end - test_start) / numer_of_matrix, 'S' )

print("test time: ", test_times)
print("test time: ", np.mean(test_times))
print("numer_of_matrix: ",  numer_of_matrix)
print("test time per matrix: ", np.mean(test_times) / numer_of_matrix)

