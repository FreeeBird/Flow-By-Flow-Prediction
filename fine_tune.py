import time
import os
import numpy as np
import torch.nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import argparse
from tools.data_process import *
from tools.early_stop import EarlyStopping
from tools.tool import *
from copy import deepcopy

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='gru', help='train model name')
parser.add_argument('--epochs', default=200, help='epochs')
parser.add_argument('--tune_epochs', default=50, help='epochs')
parser.add_argument('--train_dataset', default='geant', help='chose dataset', choices=['geant', 'abilene','nobel','germany'])
parser.add_argument('--test_dataset', default='germany', help='chose dataset', choices=['geant', 'abilene','nobel','germany'])
parser.add_argument('--gpu', default=0, help='use -1/0/1 chose cpu/gpu:0/gpu:1', choices=[-1, 0, 1])
parser.add_argument('--batch_size', '--bs', default=8192, help='batch_size')
parser.add_argument('--learning_rate', '--lr', default=0.0001, help='learning_rate')
parser.add_argument('--seq_len', default=12, help='input history length')
parser.add_argument('--pre_len', default=1, help='prediction length')
parser.add_argument('--dim_model', default=32, help='dimension of embedding vector')
parser.add_argument('--dim_attn', default=32, help='dimension of attention')
parser.add_argument('--num_heads', default=1, help='attention heads')
parser.add_argument('--val_rate', default=0.2, help='val rate')
parser.add_argument('--test_rate', default=0.2, help='test rate')
parser.add_argument('--rnn_layers', default=3, help='rnn layers')
parser.add_argument('--encoder_layers', default=3, help='encoder layers')
parser.add_argument('--dropout', default=0.5, help='dropout rate')
parser.add_argument('--early_stop', default=0, help='early stop patient epochs')
parser.add_argument('--loss', default='mse', help='loss fun',choices=['mse','mae','huber'])
parser.add_argument('--l2_loss', default=0, help='use l2 loss')
parser.add_argument('--rounds',default=3)

args = parser.parse_args()
if args.gpu == 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,0"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# param


loss_func = args.loss
# hyper param
epoch = args.epochs
batch_size = args.batch_size
lr = args.learning_rate
seq_len = args.seq_len
pre_len = args.pre_len
em_size = args.dim_model
num_head = args.num_heads
val_rate = args.val_rate
rounds = args.rounds
test_rate = args.test_rate
################# data
train_dataset = args.train_dataset
test_dataset = args.test_dataset
num_nodes = get_data_nodes(train_dataset)
num_flows = num_nodes * num_nodes
test_flow = get_data_nodes(test_dataset) ** 2
# load data
train_data = np.load(get_data_path(train_dataset))
test_data = np.load(get_data_path(test_dataset))
ALL_TEST_MSE = []
ALL_TEST_MAE = []
TUNE_TEST_MSE = []
TUNE_TEST_MAE = []
for r in range(rounds):
    early_stop = EarlyStopping(patience=args.early_stop)
    # split dataset
    train_x, train_y, val_x, val_y, test_x, test_y, max_data,tune_x, tune_y = split_dataset_cross_finetune(train_data,test_data, val_rate=val_rate,test_rate=test_rate,
                                                                                    seq_len=seq_len,predict_len=pre_len)
    # ndarray -> tensor
    train_x, train_y, = torch.from_numpy(train_x).float(), torch.from_numpy(train_y).float()
    val_x, val_y, = torch.from_numpy(val_x).float(), torch.from_numpy(val_y).float()
    test_x, test_y, = torch.from_numpy(test_x).float(), torch.from_numpy(test_y).float()
    tune_x, tune_y, = torch.from_numpy(tune_x).float(), torch.from_numpy(tune_y).float()

    # tensor dataset
    train_dataset = TensorDataset(train_x, train_y)
    val_dataset = TensorDataset(val_x, val_y)
    test_dataset = TensorDataset(test_x, test_y)
    tune_dataset = TensorDataset(tune_x, tune_y)
    # dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, pin_memory=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, pin_memory=False, num_workers=8)
    tune_loader = DataLoader(tune_dataset, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=8)

    ################# model
    model = get_model(args.model, args=args)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = get_loss_func(loss_func)
    # criterion = torch.nn.SmoothL1Loss()

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
        eval_loss = eval_loss / len(val_loader)
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
            'dict/' + model.__class__.__name__+ "_" + args.train_dataset + "_" + args.test_dataset + "_" + str(seq_len) + "-" + str(pre_len) + "_" + ts + '_dict.pkl')
    print((time_end - time_start), ' S')
    print(args)
    index = val_losses.index(np.min(val_losses))
    print(
        'min_mse:%r' % (val_losses[index]),
        'min_mae:%r' % (val_maes[index]),
    )
    ########## test ###########
    test_start = time.time()
    y_true, y_pred = [], []
    model.load_state_dict(best_model_dict)
    with torch.no_grad():
        model.eval()
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            x = x.reshape([-1,x.shape[-1]])
            y_hat = model(x)
            y = y.squeeze(-1)
            y_hat = y_hat.reshape([-1,test_flow])
            loss = criterion(y_hat, y)
            y = y.cpu().detach().numpy()
            y_hat = y_hat.cpu().detach().numpy()
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
    ALL_TEST_MAE.append(mae)
    ALL_TEST_MSE.append(mse)
    # ALL_TEST_R2.append(r2)

    ################ fine tune ###############
    model.load_state_dict(best_model_dict)
    tune_losses = []
    MIN_MSE = 1e5
    TUNE_MAE = 0
    for e in range(1, args.tune_epochs + 1):
        model.train()
        tune_loss = 0.0
        for x, y in tqdm(tune_loader):
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tune_loss += loss.item()
        tune_losses.append(tune_loss / len(tune_loader))
        # model test

        eval_loss = 0.0
        y_true = []
        y_pred = []
        with torch.no_grad():
            model.eval()
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                x = x.reshape([-1,x.shape[-1]])
                y_hat = model(x)
                y = y.squeeze(-1)
                y_hat = y_hat.reshape([-1,test_flow])
                loss = criterion(y_hat, y)
                y = y.cpu().detach().numpy()
                y_hat = y_hat.cpu().detach().numpy()
                y_pred.extend(y_hat)
                y_true.extend(y)
                eval_loss += loss.item()
        eval_loss = eval_loss / len(test_loader)
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        val_mse = mean_squared_error(y_true, y_pred)
        val_mae = mean_absolute_error(y_true, y_pred)
        val_losses.append(val_mse)
        val_maes.append(val_mae)
        if val_mse < MIN_MSE:
            MIN_MSE = val_mse
            TUNE_MAE = val_mae
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
        torch.save(best_model_dict,
        'dict/fine_' + model.__class__.__name__+ "_" + args.train_dataset + "_" + args.test_dataset + "_" + str(seq_len) + "-" + str(pre_len) + "_" + ts + '_dict.pkl')
    TUNE_TEST_MSE.append(MIN_MSE)
    TUNE_TEST_MAE.append(TUNE_MAE)
    print('ALL TEST MSE:')
    print(ALL_TEST_MSE)
    print('ALL TEST MAE:')
    print(ALL_TEST_MAE)
    print('ALL TUNE MSE:')
    print(TUNE_TEST_MSE)
    print('ALL TUNE MAE:')
    print(TUNE_TEST_MAE)

print('############# Conclusion ###########')
print('ALL TEST MSE:')
print(ALL_TEST_MSE)
print(np.mean(ALL_TEST_MSE), " ± " ,np.std(ALL_TEST_MSE))
print('ALL TEST MAE:')
print(ALL_TEST_MAE)
print(np.mean(ALL_TEST_MAE), " ± " ,np.std(ALL_TEST_MAE))
print('ALL TUNE MSE:')
print(TUNE_TEST_MSE)
print(np.mean(TUNE_TEST_MSE), " ± " ,np.std(TUNE_TEST_MSE))
print('ALL TUNE MAE:')
print(TUNE_TEST_MAE)
print(np.mean(TUNE_TEST_MAE), " ± " ,np.std(TUNE_TEST_MAE))
