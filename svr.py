'''
Author: FreeeBird
Date: 2021-12-15 20:45:53
LastEditTime: 2022-05-27 22:01:04
LastEditors: FreeeBird
Description: 
FilePath: /flow-wise-prediction/svr.py
'''
from sklearn import svm
from tools.data_process import *
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm
# import torch

fea_path = 'dataset/abilene_fea.npy'
data = np.load(fea_path)

max_data = np.max(data, axis=0)
data = col_norm(data,max_data)

pre_len = 3
test_index = int(data.shape[0] * 0.8)
train_data = data[:test_index]
test_data = data[test_index:]

train_x, train_y = matrix_sample_maker(train_data, 12, 1)
test_x, test_y = matrix_sample_maker(test_data, 12, pre_len)

train_x, train_y = np.array(train_x),np.array(train_y)
test_x, test_y = np.array(test_x),np.array(test_y).squeeze()


y_pred = np.zeros_like(test_y)
for i in tqdm(range(train_x.shape[2])):
    clf = svm.SVR()
    tx = train_x[:,:,i].squeeze()
    ty = train_y[:,i].squeeze()
    clf.fit(tx,ty)
    pred = clf.predict(test_x[:,:,i])
    y_pred[:,0,i] = pred
    if pre_len>=2:
        x = np.concatenate((test_x[:,1:,i],pred[:,None]),1)
        y2 = clf.predict(x)
        y_pred[:,1,i] = y2
    if pre_len>=3:
        x = np.concatenate((test_x[:,2:,i],pred[:,None],y2[:,None]),1)
        y3 = clf.predict(x)
        y_pred[:,2,i] = y3

mse = mean_squared_error(test_y.flatten(),y_pred.flatten())
mae = mean_absolute_error(test_y.flatten(),y_pred.flatten())
print(mse)
print(mae)

