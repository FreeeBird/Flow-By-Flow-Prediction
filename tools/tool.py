'''
Author: FreeeBird
Date: 2021-11-09 21:23:19
LastEditTime: 2022-05-31 14:57:37
LastEditors: FreeeBird
Description: 
FilePath: /flow-wise-prediction/tools/tool.py
'''
import torch
from model import *
from model.LSTNet import LSTNet
from model.convlstm import EncoderDecoderConvLSTM
from model.dcrnn import DCRNNModel
from model.gru import GRU, GRU_TM
from model.lstm import LSTM, LSTM_TM
from model.mtgnn import gtnet
# from tsai.all import * 
from model.stgcn import STGCN
from model.transformer import Transformer
from model.transformer_tm import Transformer_TM
from tsai.all import MLP,ResNet,TST,TCN

def get_data_path(dataset='geant'):
    fea_path = ''
    if dataset == 'geant':
        fea_path = '/home/liyiyong/flow-wise-prediction/dataset/geant_fea.npy'
    elif dataset == 'abilene':
        fea_path = '/home/liyiyong/flow-wise-prediction/dataset/abilene_fea.npy'
    elif dataset == 'abilene05':
        fea_path = '/home/liyiyong/flow-wise-prediction/dataset/abilene_fea_05.npy'
    elif dataset == 'nobel':
        fea_path = '/home/liyiyong/flow-wise-prediction/dataset/nobel_germany.npy'
    elif dataset == 'germany':
        fea_path = '/home/liyiyong/flow-wise-prediction/dataset/germany50.npy'
    return fea_path


def get_adj_matrix(dataset='abilene'):
    adj_path = ''
    if dataset == 'geant':
        adj_path = '/home/liyiyong/flow-wise-prediction/dataset/geant_adj.npy'
    elif dataset == 'abilene':
        adj_path = '/home/liyiyong/flow-wise-prediction/dataset/abilene_adj.npy'
    return adj_path

def get_data_nodes(dataset='geant'):
    nodes = 0
    if dataset == 'geant':
        nodes = 23
    elif dataset == 'abilene':
        nodes = 12
    elif dataset == 'abilene05':
        nodes = 12
    elif dataset == 'nobel':
        nodes = 17
    elif dataset == 'germany':
        nodes = 50
    return nodes

def get_loss_func(loss = 'mse'):
    if loss == 'mse':
        return torch.nn.MSELoss()
    if loss == 'mae':
        return torch.nn.L1Loss()
    if loss == 'huber':
        return torch.nn.SmoothL1Loss()

def get_model(name='lstm', args=None):
    model = None
    if name == 'lstm':
        model = LSTM(in_dim=1, hidden_dim=args.dim_model, n_layer=args.rnn_layers, seq_len=args.seq_len, pre_len=args.pre_len)
    if name == 'lstm_tm':
        model = LSTM_TM(in_dim=args.num_flows, hidden_dim=args.dim_model, n_layer=args.rnn_layers, seq_len=args.seq_len, pre_len=args.pre_len)
    if name == 'gru':
        model = GRU(in_dim=1, hidden_dim=args.dim_model, n_layer=args.rnn_layers, seq_len=args.seq_len,
                    pre_len=args.pre_len,
                    dropout=args.dropout)
    if name == 'gru_tm':
        model = GRU_TM(in_dim=args.num_flows, hidden_dim=args.dim_model, n_layer=args.rnn_layers, seq_len=args.seq_len,
                    pre_len=args.pre_len,
                    dropout=args.dropout)
    if name == 'transformer':
        model = Transformer(in_channels=1, embed_size=args.dim_model, seq_len=args.seq_len,
                                pre_len=args.pre_len, heads=args.num_heads,n_encoder_layers=args.encoder_layers)
    if name == 'transformer_tm':
        model = Transformer_TM(in_channels=args.num_flows, embed_size=args.dim_model, seq_len=args.seq_len,
                                pre_len=args.pre_len, heads=args.num_heads,n_encoder_layers=args.encoder_layers)
    if name == 'dcrnn':
        model = DCRNNModel(args.m_adj,args.seq_len,args.num_nodes,args.pre_len)
    if name == 'stgcn':
        model = STGCN(args.num_nodes,args.num_nodes,args.seq_len,args.pre_len)
    if name == 'mlp':
        model = MLP(args.num_flows,args.num_flows,args.seq_len)
    if name == 'resnet':
        model = ResNet(args.seq_len,args.num_flows)
    if name =='tcn':
        model = TCN(args.seq_len,args.num_flows,fc_dropout=.5)
    if name == 'convlstm_tm':
        model = EncoderDecoderConvLSTM(args.dim_model,1)
    if name == 'LSTNet':
        model = LSTNet(flows=args.num_flows,seq_len=args.seq_len,pre_len=args.pre_len,hidCNN=args.dim_model,hidRNN=args.dim_model,hidSkip=10,CNN_kernel=6,skip=2)
    # if name == 'Rocket':
    #     model = ROCKET(args.num_flows,args.seq_len)
    if name == 'TST':
        model = TST(args.num_flows, args.num_flows, args.seq_len, max_seq_len=args.seq_len)
    if name == 'mtgnn':
        model = gtnet(gcn_true=True, buildA_true=True, gcn_depth=2, num_nodes=args.num_nodes, device=args.device, predefined_A=args.m_adj, static_feat=None, 
    dropout=args.dropout, subgraph_size=2, node_dim=args.dim_model, dilation_exponential=1, conv_channels=args.dim_model, residual_channels=args.dim_model, 
    skip_channels=args.dim_model, end_channels=args.dim_model, seq_length=args.seq_len, in_dim=args.num_nodes, out_dim=args.num_nodes, layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True)
    return model
