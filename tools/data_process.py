import numpy as np

from sklearn.model_selection import train_test_split


def col_norm(data, col_max=None):
    if col_max is None:
        col_max = np.max(data, axis=0)
        # col_min = np.min(data,axis=0)
    data = data / col_max
    data[np.isnan(data)] = 0.0
    data[np.isinf(data)] = 0.0
    return data


def max_normalization(data):
    return data / np.max(data)


def inverse_normalization(data, col_max, col_min):
    return data * (col_max - col_min) + col_min


def flow_sample_maker(data, seq_len=3, pre_len=1,train=True):
    # assert data = [time,flows]
    # data = [flows,time]
    data = data.T
    x, y = [], []
    time_len = data.shape[1]
    for i in range(time_len - seq_len - pre_len + 1):
        sample = data[:, i:i + seq_len + pre_len]
        x.append(sample[:, 0:seq_len])
        y.append(sample[:, seq_len:seq_len + pre_len])
    x, y = np.array(x), np.array(y)
    if train:
        x = x.reshape(-1,x.shape[-1])
        y = y.reshape(-1,y.shape[-1])
    return x, y


def matrix_sample_maker(data, seq_len=3, pre_len=1):
    x, y = [], []
    time_len = data.shape[0]
    for i in range(time_len - seq_len - pre_len + 1):
        sample = data[i:i + seq_len + pre_len]
        x.append(sample[0:seq_len])
        if(pre_len>1):
            y.append(sample[seq_len:seq_len + pre_len])
        else:
            y.extend(sample[seq_len:seq_len + pre_len])
    x, y = np.array(x), np.array(y)
    return x, y


def split_dataset(data, train_rate = 0.7,val_rate=0.1, seq_len=12, predict_len=3, wise='flow'):
    max_data = np.max(data, axis=0)
    data = col_norm(data,max_data)
    # data = max_normalization(data)
    test_index = int(data.shape[0] * (train_rate + val_rate))
    test_data = data[test_index:]
    val_index = int(data.shape[0] * train_rate)
    data = data[:test_index]
    # split train val
    train_data = data[:val_index]
    val_data = data[val_index:]
    if wise == 'flow':
        train_x, train_y = flow_sample_maker(train_data, seq_len, predict_len)
        val_x, val_y = flow_sample_maker(val_data, seq_len, predict_len,False)
        test_x, test_y = flow_sample_maker(test_data, seq_len, predict_len,False)
    else:
        train_x, train_y = matrix_sample_maker(train_data, seq_len, predict_len)
        val_x, val_y = matrix_sample_maker(val_data, seq_len, predict_len)
        test_x, test_y = matrix_sample_maker(test_data, seq_len, predict_len)
    return train_x, train_y, val_x, val_y, test_x, test_y, max_data


def random_split_dataset(data, train_rate = 0.7, val_rate=0.1, seq_len=12, predict_len=1):
    data = col_norm(data)
    max_data = np.max(data, axis=0)
    # min_data = np.max(data, axis=0)
    x, y = flow_sample_maker(data, seq_len, predict_len,False)
    # array -> np array
    x, y = np.array(x), np.array(y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=(train_rate+val_rate))
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=(train_rate / (train_rate+val_rate)))

    return x_train, y_train, x_val, y_val, x_test, y_test, max_data


def matrix_2_flow(data):
    if len(data.shape) < 3:
        data = np.expand_dims(data, 1)
    N,T,F = data.shape
    data = data.transpose(0,2,1).reshape([N*F,T])
    return data

def random_split_dataset_by_matrix(data, train_rate = 0.7, val_rate=0.1, seq_len=12, predict_len=1):
    data = col_norm(data)
    max_data = np.max(data, axis=0)
    # min_data = np.max(data, axis=0)
    x, y = matrix_sample_maker(data, seq_len, predict_len)
    # array -> np array
    x, y = np.array(x), np.array(y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=(train_rate+val_rate))
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=(train_rate / (train_rate+val_rate)))
    x_train, y_train, x_val, y_val = matrix_2_flow(x_train), matrix_2_flow(y_train), matrix_2_flow(x_val), matrix_2_flow(y_val)
    x_test, y_test = matrix_2_flow(x_test), matrix_2_flow(y_test)
    return x_train, y_train, x_val, y_val, x_test, y_test, max_data


def split_dataset_sample_time(data,train_rate = 0.7, val_rate=0.1, seq_len=12, predict_len=3, sample_rate=0.5):
    max_data = np.max(data, axis=0)
    data = col_norm(data,max_data)
    test_index = int(data.shape[0] * (train_rate+val_rate))
    test_data = data[test_index:]
    val_index = int(data.shape[0] * train_rate)
    data = data[:test_index]
    # split train val
    train_data = data[:val_index]
    val_data = data[val_index:]

    sample_len = int(train_data.shape[0] * sample_rate)
    
    train_index = int(np.random.choice(np.arange(train_data.shape[0]-sample_len), replace=False)) 
    train_data = train_data[train_index:train_index+sample_len]

    train_x, train_y = flow_sample_maker(train_data, seq_len, predict_len)
    val_x, val_y = flow_sample_maker(val_data, seq_len, predict_len,False)
    test_x, test_y = flow_sample_maker(test_data, seq_len, predict_len,False)
    return train_x, train_y, val_x, val_y, test_x, test_y, max_data,train_index,train_index+sample_len



def matrix_sample_time(data,train_rate = 0.7, val_rate=0.1, seq_len=12, predict_len=1, sample_rate=0.5):
    max_data = np.max(data, axis=0)
    data = col_norm(data,max_data)
    test_index = int(data.shape[0] * (train_rate+val_rate))
    test_data = data[test_index:]
    val_index = int(data.shape[0] * train_rate)
    data = data[:test_index]
    # split train val
    train_data = data[:val_index]
    val_data = data[val_index:]

    sample_len = int(train_data.shape[0] * sample_rate)
    
    train_index = int(np.random.choice(np.arange(train_data.shape[0]-sample_len), replace=False)) 
    train_data = train_data[train_index:train_index+sample_len]

    train_x, train_y = matrix_sample_maker(train_data, seq_len, predict_len)
    val_x, val_y = matrix_sample_maker(val_data, seq_len, predict_len)
    test_x, test_y = matrix_sample_maker(test_data, seq_len, predict_len)
    return train_x, train_y, val_x, val_y, test_x, test_y, max_data,train_index,train_index+sample_len

def split_dataset_sample_flow(data, train_rate = 0.7, val_rate=0.1, seq_len=12, predict_len=3, sample_rate=0.5):
    max_data = np.max(data, axis=0)
    data = col_norm(data,max_data)
    test_index = int(data.shape[0] * (train_rate+val_rate))
    test_data = data[test_index:]
    val_index = int(data.shape[0] * train_rate)
    data = data[:test_index]
    # split train val
    train_data = data[:val_index]
    val_data = data[val_index:]
    # sample
    sample_len = train_data.shape[1]
    if sample_rate > 0:
        sample_num = int(sample_len * sample_rate)
    else:
        sample_num = 1
    train_index = np.random.choice(np.arange(sample_len), sample_num,replace=False)
    train_index = np.sort(train_index)
    print("sample flow index: ",train_index)
    train_data = train_data[:, train_index]

    train_x, train_y = flow_sample_maker(train_data, seq_len, predict_len)
    val_x, val_y = flow_sample_maker(val_data, seq_len, predict_len,False)
    test_x, test_y = flow_sample_maker(test_data, seq_len, predict_len,False)
    return train_x, train_y, val_x, val_y, test_x, test_y, max_data,train_index

def split_dataset_sample(data, train_rate = 0.7, val_rate=0.1, seq_len=12, predict_len=3, sample_rate=0.5):
    max_data = np.max(data, axis=0)
    data = col_norm(data,max_data)
    test_index = int(data.shape[0] * (train_rate+val_rate))
    test_data = data[test_index:]
    val_index = int(data.shape[0] * train_rate)
    data = data[:test_index]
    # split train val
    train_data = data[:val_index]
    val_data = data[val_index:]
   
   
    
    

    train_x, train_y = flow_sample_maker(train_data, seq_len, predict_len)
     # sample
    sample_len = train_x.shape[0]
    if sample_rate > 0:
        sample_num = int(sample_len * sample_rate)
    else:
        sample_num = 1
    train_index = np.random.choice(np.arange(sample_len), sample_num)
    train_x = train_x[train_index]
    train_y = train_y[train_index]
    val_x, val_y = flow_sample_maker(val_data, seq_len, predict_len,False)
    test_x, test_y = flow_sample_maker(test_data, seq_len, predict_len,False)
    return train_x, train_y, val_x, val_y, test_x, test_y, max_data




def split_dataset_cross(train_data,test_data,val_rate = 0.2,test_rate = 0.2, seq_len=12, predict_len=1):
    max_data = np.max(train_data, axis=0)
    train_data = col_norm(train_data,max_data)
    val_index = int(train_data.shape[0] * (1-val_rate))
    # split train val
    val_data = train_data[val_index:]
    train_data = train_data[:val_index]

    test_data = col_norm(test_data)
    if test_rate>0:
        test_data = test_data[int(test_data.shape[0]*(1-test_rate)):]
    train_x, train_y = flow_sample_maker(train_data, seq_len, predict_len)
    val_x, val_y = flow_sample_maker(val_data, seq_len, predict_len,False)
    test_x, test_y = flow_sample_maker(test_data, seq_len, predict_len,False)
    return train_x, train_y, val_x, val_y, test_x, test_y, max_data


def split_dataset_cross_finetune(train_data,test_data,val_rate = 0.2,test_rate = 0.2, seq_len=12, predict_len=1):
    max_data = np.max(train_data, axis=0)
    train_data = col_norm(train_data,max_data)
    val_index = int(train_data.shape[0] * (1-val_rate))
    # split train val
    val_data = train_data[val_index:]
    train_data = train_data[:val_index]

    test_data = col_norm(test_data)
    test_index = int(test_data.shape[0]*(1-test_rate))
    tune_data = test_data[:test_index]
    test_data = test_data[test_index:]
    train_x, train_y = flow_sample_maker(train_data, seq_len, predict_len)
    tune_x, tune_y = flow_sample_maker(tune_data, seq_len, predict_len)
    val_x, val_y = flow_sample_maker(val_data, seq_len, predict_len,False)
    test_x, test_y = flow_sample_maker(test_data, seq_len, predict_len,False)
    return train_x, train_y, val_x, val_y, test_x, test_y, max_data,tune_x, tune_y



def split_dataset_cross_sample_flow(train_data,test_data,val_rate = 0.2,test_rate = 0.2, seq_len=12, predict_len=1,sample_rate=0.5):
    max_data = np.max(train_data, axis=0)
    train_data = col_norm(train_data,max_data)
    val_index = int(train_data.shape[0] * (1-val_rate))
    # split train val
    val_data = train_data[val_index:]
    train_data = train_data[:val_index]
    sample_len = train_data.shape[1]
    if sample_rate > 0:
        sample_num = int(sample_len * sample_rate)
    else:
        sample_num = 1
    train_index = np.random.choice(np.arange(sample_len), sample_num,replace=False)
    train_index = np.sort(train_index)
    print("sample flow index: ",train_index)
    train_data = train_data[:, train_index]
    # test_data
    test_data = col_norm(test_data)
    if test_rate>0:
        test_data = test_data[int(test_data.shape[0]*(1-test_rate)):]
    train_x, train_y = flow_sample_maker(train_data, seq_len, predict_len)
    val_x, val_y = flow_sample_maker(val_data, seq_len, predict_len,False)
    test_x, test_y = flow_sample_maker(test_data, seq_len, predict_len,False)
    return train_x, train_y, val_x, val_y, test_x, test_y, max_data


def split_dataset_cross_sample_time(train_data,test_data,val_rate = 0.2,test_rate = 0.2, seq_len=12, predict_len=1,sample_rate=0.5):
    max_data = np.max(train_data, axis=0)
    train_data = col_norm(train_data,max_data)
    val_index = int(train_data.shape[0] * (1-val_rate))
    # split train val
    val_data = train_data[val_index:]
    train_data = train_data[:val_index]
    sample_len = int(train_data.shape[0] * sample_rate)
    train_index = int(np.random.choice(np.arange(train_data.shape[0]-sample_len), replace=False)) 
    train_data = train_data[train_index:train_index+sample_len]

    print("sample flow index: ",train_index)
    # test_data
    test_data = col_norm(test_data)
    if test_rate>0:
        test_data = test_data[int(test_data.shape[0]*(1-test_rate)):]
    train_x, train_y = flow_sample_maker(train_data, seq_len, predict_len)
    val_x, val_y = flow_sample_maker(val_data, seq_len, predict_len,False)
    test_x, test_y = flow_sample_maker(test_data, seq_len, predict_len,False)
    return train_x, train_y, val_x, val_y, test_x, test_y, max_data,train_index,train_index+sample_len