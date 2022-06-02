import torch
import torch.nn as nn
import numpy as np


class Attention(nn.Module):
    def __init__(self, embed_size, heads):
        super(Attention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        assert (
                self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.W_V = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_K = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_Q = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, input_Q, input_K, input_V):
        B, T, C = input_Q.shape
        # [B, T, C] --> [B, T, h * d_k] --> [B, T, h, d_k] --> [B, h, T, d_k]
        Q = self.W_Q(input_Q).view(B, T, self.heads, self.head_dim).permute(0, 2, 1, 3)  # Q: [B, h, T, d_k]
        K = self.W_K(input_K).view(B, T, self.heads, self.head_dim).permute(0, 2, 1, 3)  # K: [B, h,  T, d_k]
        V = self.W_V(input_V).view(B, T, self.heads, self.head_dim).permute(0, 2, 1, 3)  # V: [B, h,  T, d_k]

        context = ScaledDotProductAttention()(Q, K, V)  # [B, h,  T, d_k]
        context = context.permute(0, 2, 1, 3)  # [B,T, h, d_k]
        context = context.reshape(B, T, self.heads * self.head_dim)  # [B, N, T, C]
        output = self.fc_out(context)  # [batch_size, len_q, d_model]
        return output


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V):
        B, n_heads, len2, d_k = Q.shape
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context


class Encoder(nn.Module):

    def __init__(self, d_model, dim_ff=2048, dropout=0.5, heads=1):
        super(Encoder, self).__init__()
        self.attention = Attention(d_model, heads)
        self.linear1 = nn.Linear(d_model, dim_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_ff, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, src):
        src2 = self.attention(src, src, src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src



class Decoder(nn.Module):

    def __init__(self, d_model, dim_ff=2048, dropout=0.3,heads=1):
        super(Decoder,self).__init__()
        self.attention = Attention(d_model, heads)
        self.linear1 = nn.Linear(d_model, dim_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_ff, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    #  the queries come from the previous decoder layer,
    # and the memory keys and values come from the output of the encoder
    def forward(self, tgt, src): # src, en_src
        tgt2 = self.attention(tgt, tgt, tgt)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.attention(tgt, src, src)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class Transformer_TM(nn.Module):
    def __init__(
            self,
            in_channels=1,
            embed_size=128,
            seq_len=12,
            pre_len=1,
            heads=1,
            dropout=0.5,
            dim_ff=1024,
            n_encoder_layers=3,
            n_decoder_layers=3,
            positional_embedding='fixed'
    ):
        super(Transformer_TM, self).__init__()
        self.embedding = nn.Linear(in_channels, embed_size)
        self.encs = nn.ModuleList()
        for i in range(n_encoder_layers):
            self.encs.append(
                Encoder(embed_size, heads=heads, dropout=dropout, dim_ff=dim_ff)
            )
        # self.decs = nn.ModuleList()
        # for i in range(n_decoder_layers):
        #     self.decs.append(
        #         Decoder(embed_size,dim_ff=dim_ff,heads=heads,dropout=dropout)
        #     )
        # self.encoder = Encoder(embed_size, heads=heads, dropout=dropout, dim_ff=dim_ff)
        self.time_linear = nn.Linear(seq_len, pre_len)
        self.final_layer = nn.Linear(embed_size, in_channels)
        # self.position_embedding = get_positional_embedding(positional_embedding, embed_size, seq_len, dropout)
        self.act = nn.ReLU()
        # self.LearnEncoding = LearnEncoding(embed_size,T_dim)

    def forward(self, x):
        # BS,T,F = x.shape
        x = self.embedding(x)  # BS,T,es
        # x = self.LearnEncoding(x)
        # x = self.position_embedding(x)
        # x = self.encoder(x)  # BS,T,es
        x = self.encs[0](x)
        for enc in self.encs[1:]:
            x = enc(x)
        # x = self.decs[0](x,en_src)
        # for dec in self.decs[1:]:
        #     x = dec(x,en_src)

        x = self.final_layer(x)  # BS,t,1
        # x = self.act(x)
        return x[:,-1,] # [B, pre_len, f]


def get_positional_embedding(pos_em='fixed', d_model=64, max_len=12, dropout=0.5):
    if pos_em == 'fixed':
        return PositionalEncoding(d_model, max_len, dropout)
    if pos_em == 'learnable':
        return LearnablePositionalEncoding(d_model, dropout, max_len)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model=64, max_len=12, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.pe = torch.zeros(max_len, d_model).to(device='cuda')
        self.pe.requires_grad = False
        self.dropout = nn.Dropout(dropout)
        pos = torch.arange(max_len).float().unsqueeze(1)
        _2i = torch.arange(0, d_model, step=2).float()
        self.pe[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.pe[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    def forward(self, x):
        x = self.dropout(x + self.pe)
        return x


class LearnablePositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Each position gets its own embedding
        # Since indices are always 0 ... max_len, we don't have to do a look-up
        self.pe = nn.Parameter(torch.empty(max_len, 1, d_model))  # requires_grad automatically set to True
        nn.init.uniform_(self.pe, 0, 1)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """
        x = x.permute(1, 0, 2)
        x = x + self.pe[:x.size(0), :]
        x = x.permute(1, 0, 2)
        return self.dropout(x)
