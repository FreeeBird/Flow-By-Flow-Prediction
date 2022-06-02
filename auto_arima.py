'''
Author: FreeeBird
Date: 2021-12-14 21:54:54
LastEditTime: 2022-05-31 22:07:51
LastEditors: FreeeBird
Description: 
FilePath: /flow-wise-prediction/auto_arima.py
'''
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima_model import ARIMA
import warnings
from tools.data_process import col_norm

from tools.tool import get_data_path
warnings.filterwarnings("ignore")#不显示警告信息
import pmdarima as pm
from pmdarima import model_selection
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
###### evaluation ######
def evaluation(a, b):
    mse = mean_squared_error(a, b)
    mae = mean_absolute_error(a, b)
    return mse,mae



def sample_maker(data, seq_len=3, pre_len=1):
    data = data.T
    time_len = data.shape[1]
    samples = []
    for i in range(time_len//(seq_len+pre_len)):
        i = i*(seq_len+pre_len)
        sample = data[:, i:i + seq_len + pre_len]
        samples.append(sample)
    samples = np.array(samples)
    # samples = samples.reshape(-1,samples.shape[-1])
    # samples = samples[:,3]
    return samples


seq_len = 12
predict_len = 3
dataset = 'geant'
num_flows = 529
fea_path = get_data_path(dataset)
data = np.load(fea_path)
max_data = np.max(data, axis=0)
data = col_norm(data,max_data)
test_index = int(data.shape[0] * 0.8)
test_data = data[test_index:]
train_data = data[int(test_index-(30*24*60/15)):test_index]
pre_len = test_data.shape[0]
t1,t2,t3 = [],[],[]
p1,p2,p3 = [],[],[]
amse,amae=[],[]
amse2,amae2=[],[]
amse3,amae3=[],[]
for i in tqdm(range(num_flows)):
    train = train_data[:,i]
    test = test_data[:,i]
    arima = pm.auto_arima(train, error_action='ignore', trace=False,
                    suppress_warnings=False, maxiter=2,
                    seasonal=False, m=1)
    pred = arima.predict(n_periods=pre_len)
    mse,mae = evaluation(test[:-2],pred[:-2])
    mse2,mae2 = evaluation(test[1:-1],pred[1:-1])
    mse2,mae2 = mse2+mse,mae+mae2
    mse3,mae3 = evaluation(test[2:],pred[2:])
    mse3,mae3 = mse2+mse3,mae3+mae2
    amse.append(mse)
    amae.append(mae)
    amse2.append(mse2)
    amae2.append(mae2)
    amse3.append(mse3)
    amae3.append(mae3)
    # pred2 = arima.predict(n_periods=2)
    # pred3 = arima.predict(n_periods=3)

mse,mae = np.mean(amse),np.mean(amae)
mse2,mae2 = np.mean(amse2),np.mean(amae2)
mse3,mae3 = np.mean(amse3),np.mean(amae3)
print(mse,mae)
print(mse2,mae2)
print(mse3,mae3)

    