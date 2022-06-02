from copy import deepcopy
import time
import os
import numpy as np
import torch.nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error
import argparse
from tools.data_process import *
from tools.early_stop import EarlyStopping
from tools.tool import *

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='transformer', help='train model name')
parser.add_argument('--seed', default=42, help='random seed')
parser.add_argument('--epochs', default=200, help='epochs')
parser.add_argument('--dataset', default='geant', help='chose dataset', choices=['geant', 'abilene'])
parser.add_argument('--gpu', default=0, help='use -1/0/1 chose cpu/gpu:0/gpu:1', choices=[-1, 0, 1])
parser.add_argument('--batch_size', '--bs', default=16*529, help='batch_size')
parser.add_argument('--learning_rate', '--lr', default=0.0001, help='learning_rate')
parser.add_argument('--seq_len', default=12, help='input history length')
parser.add_argument('--pre_len', default=1, help='prediction length')
parser.add_argument('--dim_model', default=32, help='dimension of embedding vector')
parser.add_argument('--dim_attn', default=32, help='dimension of attention')
parser.add_argument('--num_heads', default=1, help='attention heads')
parser.add_argument('--train_rate', default=0.7, help='')
parser.add_argument('--rnn_layers', default=3, help='rnn layers')
parser.add_argument('--encoder_layers', default=3, help='encoder layers')
parser.add_argument('--decoder_layers', default=1, help='decoder layers')
parser.add_argument('--dropout', default=0.5, help='dropout rate')
parser.add_argument('--early_stop', default=15, help='early stop patient epochs')
parser.add_argument('--loss', default='mse', help='loss fun',choices=['mse','mae','huber'])
parser.add_argument('--l2_loss', default=0, help='use l2 loss')
parser.add_argument('--rounds',default=5)
parser.add_argument('--sample_rate',default=0.3)

args = parser.parse_args()
if args.gpu == 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,0"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# param
dataset = args.dataset
fea_path = get_data_path(dataset)
num_nodes = get_data_nodes(dataset)
num_flows = num_nodes * num_nodes

loss_func = args.loss
# hyper param
epoch = args.epochs
batch_size = args.batch_size
lr = args.learning_rate
seq_len = args.seq_len
pre_len = args.pre_len
em_size = args.dim_model
num_head = args.num_heads
train_rate = args.train_rate
rounds = args.rounds
sample_rate = args.sample_rate
################# data
# load data
data = np.load(fea_path)
ALL_TEST_MSE = []
ALL_TEST_MAE = []
ALL_PRE_TIME = []
ALL_TRAIN_INDEX = []
for r in range(rounds):
    early_stop = EarlyStopping(patience=args.early_stop)
    # split dataset
    train_x, train_y, val_x, val_y, test_x, test_y, max_data,train_index = split_dataset_sample_flow(data, train_rate=train_rate,val_rate=0.1,
                                                                                    seq_len=seq_len,
                                                                                    predict_len=pre_len,sample_rate=sample_rate)
    ALL_TRAIN_INDEX.append(train_index)
    # ndarray -> tensor
    train_x, train_y, = torch.from_numpy(train_x).float(), torch.from_numpy(train_y).float()
    val_x, val_y, = torch.from_numpy(val_x).float(), torch.from_numpy(val_y).float()
    test_x, test_y, = torch.from_numpy(test_x).float(), torch.from_numpy(test_y).float()

    # tensor dataset
    train_dataset = TensorDataset(train_x, train_y)
    val_dataset = TensorDataset(val_x, val_y)
    test_dataset = TensorDataset(test_x, test_y)
    # dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, pin_memory=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, pin_memory=False, num_workers=8)

    ################# model
    model = get_model(args.model, args=args)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = get_loss_func(loss_func)

    print(args)

    # test_y = test_y*max_data
    time_start = time.time()
    train_losses = []
    val_losses, val_maes, = [], []
    MIN_MSE = 1e5
    EPOCH = 1
    best_model_dict = deepcopy(model.state_dict())
    ###### train ####
    for e in range(1, epoch + 1):
        model.train()
        train_loss = 0.0
        for x, y in tqdm(train_loader):
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_losses.append(train_loss / len(train_loader))
        # model val

        eval_loss = 0.0
        y_true = []
        y_pred = []
        with torch.no_grad():
            model.eval()
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                x = x.reshape([-1,x.shape[-1]])
                y_hat = model(x)
                y = y.squeeze(-1)
                y_hat = y_hat.reshape([-1,num_flows])
                loss = criterion(y_hat, y)
                y = y.cpu().detach().numpy()
                y_hat = y_hat.cpu().detach().numpy()
                y_pred.extend(y_hat)
                y_true.extend(y)
                eval_loss += loss.item()
        eval_loss = eval_loss / len(test_loader)
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        # print(test_loss)
        # mse,ame,rmse,r2,mape
        val_mse = mean_squared_error(y_true, y_pred)
        val_mae = mean_absolute_error(y_true, y_pred)
        val_losses.append(val_mse)
        val_maes.append(val_mae)
        if val_mse < MIN_MSE:
            MIN_MSE = val_mse
            best_model_dict = deepcopy(model.state_dict())
            EPOCH = e
            print('*MIN VAL LOSS:{:.6} at epoch {}'.format(MIN_MSE, EPOCH))
        if early_stop(val_mse):
            break
        print('Epoch:{}'.format(e),
            'train_mse:{:.6}'.format(train_losses[-1]),
            'val_mse:{:.6}'.format(val_losses[-1]),
            'val_mae:{:.6}'.format(val_maes[-1]),
            )

    time_end = time.time()
    ts = time.strftime("%m-%d_%H:%M:%S", time.localtime())
    print(ts)
    torch.save(best_model_dict,
            'dict/' + model.__class__.__name__+ "_" + args.dataset + "_" + str(seq_len) + "-" + str(pre_len) + "_" + ts + '_dict.pkl')
    print((time_end - time_start) / 3600, 'h')
    print(args)
    index = val_losses.index(np.min(val_losses))
    print(
        'min_mse:%r' % (val_losses[index]),
        'min_mae:%r' % (val_maes[index]),
    )
    ########## test ###########
    test_start = time.time()
    model.load_state_dict(best_model_dict)
    y_true, y_pred = [], []
    pre_time = []
    with torch.no_grad():
        model.eval()
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            x = x.reshape([-1,x.shape[-1]])
            temp_time = time.time()
            y_hat = model(x)
            temp_time = time.time() - temp_time
            y = y.squeeze(-1)
            y_hat = y_hat.reshape([-1,num_flows])
            loss = criterion(y_hat, y)
            y = y.cpu().detach().numpy()
            y_hat = y_hat.cpu().detach().numpy()
            pre_time.append(temp_time / y.shape[0])
            y_pred.extend(y_hat)
            y_true.extend(y)
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    print('TEST RESULT:',
        'mse:{:.6}'.format(mse),
        'mae:{:.6}'.format(mae),
        )
    test_end = time.time()
    print("test time: ", (time_end - time_start), ' S')
    print("predict time for single matrix:",np.mean(pre_time), 'S' )
    ALL_PRE_TIME.append(np.mean(pre_time))
    ALL_TEST_MAE.append(mae)
    ALL_TEST_MSE.append(mse)

print('############# Conclusion ###########')
print('ALL TEST MSE:')
print(ALL_TEST_MSE)
print(np.mean(ALL_TEST_MSE), " ± " ,np.std(ALL_TEST_MSE))
print('ALL TEST MAE:')
print(ALL_TEST_MAE)
print(np.mean(ALL_TEST_MAE), " ± " ,np.std(ALL_TEST_MAE))
print('ALL SAMPLE FLOWS:')
print(ALL_TRAIN_INDEX)
print('ALL TEST PRE TIME:')
print(ALL_PRE_TIME)
print(np.mean(ALL_PRE_TIME), " ± " ,np.std(ALL_PRE_TIME))
