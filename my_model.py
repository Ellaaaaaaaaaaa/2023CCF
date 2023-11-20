from torch import nn


class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, dropout, bias=False):
        super(GraphConvolution, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        nn.init.xavier_uniform_(self.weight)  # xavier初始化，就是论文里的glorot初始化
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)

    def forward(self, inputs, adj):
        # inputs: (N, n_channels), adj: sparse_matrix (N, N)
        support = torch.mm(self.dropout(inputs), self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GCN(nn.Module):
    def __init__(self, n_features, hidden_dim, dropout, n_classes):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(n_features, hidden_dim, dropout)
        self.gc2 = GraphConvolution(hidden_dim, n_classes, dropout)
        self.relu = nn.ReLU()

    def forward(self, inputs, adj):
        x = inputs
        x = self.relu(self.gc1(x, adj))
        x = self.gc2(x, adj)
        return x


# %%

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(in_features, out_features))
        self.a  = nn.Parameter(torch.zeros(2 * out_features, 1))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        '''
        h: (N, in_features)
        adj: sparse matrix with shape (N, N)
        p
        '''
        adj=torch.squeeze(adj,-1)
        # torch.Size([4, 1140, 35])

        Wh = torch.matmul(h, self.W)  # (N, out_features)

        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])  # (N, 1)
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])  # (N, 1)
        # torch.Size([4, 1140, 1])
        # torch.Size([4, 1140, 1])

        # Wh1 + Wh2.T 是N*N矩阵，第i行第j列是Wh1[i]+Wh2[j]
        # 那么Wh1 + Wh2.T的第i行第j列刚好就是文中的a^T*[Whi||Whj]
        # e矩阵 代表着节点i对节点j的attention
        e = self.leakyrelu(Wh1 +torch.transpose(Wh2,2,1))  # (N, N)
        # torch.Size([4, 1, 1140])
        # padding 是一个与 e 形状相同的矩阵，其中的所有元素都是一个很小的负数。
        # 以便在执行下一步的 mask 操作时，将注意力矩阵中的某些位置置为负无穷，使其在 softmax 操作中趋近于零
        # 即邻接矩阵中没有边相连的位置
        padding = (-2 ** 31) * torch.ones_like(e)  # (N, N)
        # adj.shape torch.Size([4, 1140, 1140])
        # padding.shape torch.Size([4, 1140, 1140])
        attention = torch.where(adj > 0, e, padding)  # (N, N)
        attention = F.softmax(attention, dim=1)  # (N, N)
        # attention矩阵第i行第j列代表node_i对node_j的注意力
        # 对注意力权重也做dropout（如果经过mask之后，attention矩阵也许是高度稀疏的，这样做还有必要吗？）
        attention = F.dropout(attention, self.dropout, training=self.training)

        h_prime = torch.matmul(attention, Wh)  # (N, out_features)
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

# nhid隐藏层输入的是边的邻接矩阵关系
# 这里没有考虑边上的值
class GAT(nn.Module):
    def __init__(self,date_emb, nfeat, nhid, dropout, alpha, nheads):
        super(GAT, self).__init__()
        date_index_number,date_dim = date_emb[0], date_emb[1]
        self.dropout = dropout
        self.MH = nn.ModuleList([
            GraphAttentionLayer(nfeat, nhid, dropout, alpha, concat=True)
            for _ in range(nheads)
        ])
        self.out_att = GraphAttentionLayer(nhid * nheads, nhid, dropout, alpha, concat=False)
        self.date_embdding = nn.Embedding(date_index_number,date_dim)
        self.active_index = nn.Linear(nhid,1)
        self.consume_index = nn.Linear(nhid,1)
    def forward(self,x_date,x_feature,x_mask_data):


        x = x_feature
        # ([4, 1140, 35])
        # x = F.dropout(x_feature, self.dropout, training=self.training)  # (N, nfeat)
        x = torch.cat([head(x, x_mask_data) for head in self.MH], dim=-1)  # (N, nheads*nhid)
        x = F.dropout(x, self.dropout, training=self.training)  # (N, nfeat)


        # x = F.dropout(x, self.dropout, training=self.training)  # (N, nheads*nhid)
        x = self.out_att(x, x_mask_data)
        # print(x.shape,x.dtype)
        act_pre= self.active_index(x)
        con_pre = self.consume_index(x)
        # torch.Size([4, 1140, 1])
        return  act_pre,con_pre


class BILSTM(nn.Module):
    def __init__(self,date_emb, nfeat, nhid, dropout, alpha, nheads):
        super(BILSTM, self).__init__()
        date_index_number,date_dim = date_emb[0], date_emb[1]
        self.dropout = dropout
        self.lstm = nn.LSTM(nfeat,
                nhid,
                num_layers=2,
                bias=True,
                batch_first=False,
                dropout=0,
                bidirectional=True)

        self.active_index = nn.Linear(2*nhid, 1)
        self.consume_index = nn.Linear(2*nhid, 1)
    def forward(self,x_date,x_feature,x_mask_data):
        lstm_out, (hidden, cell) = self.lstm(x_feature)
        x = lstm_out
        # print(x.shape)


        x = F.dropout(x, self.dropout, training=self.training)  # (N, nheads*nhid)
        act_pre= self.active_index(x)
        con_pre = self.consume_index(x)
        # print(act_pre.shape,con_pre.shape)
        return  act_pre,con_pre

### todo 新写的GAT+LSTM模型，垃圾垃圾垃圾无敌辣鸡
class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat):
        super(GATLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(in_features, out_features))
        self.a = nn.Parameter(torch.zeros(2 * out_features, 1))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        adj = torch.squeeze(adj, -1)
        Wh = torch.matmul(h, self.W)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        e = self.leakyrelu(Wh1 + torch.transpose(Wh2, 2, 1))
        padding = (-2 ** 31) * torch.ones_like(e)
        attention = torch.where(adj > 0, e, padding)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

class GATBiLSTM(nn.Module):
    def __init__(self, date_emb, nfeat, nhid_gat, nhid_lstm, dropout, alpha, nheads):
        super(GATBiLSTM, self).__init__()
        date_index_number, date_dim = date_emb[0], date_emb[1]
        self.dropout = dropout

        # GAT layers
        self.gat_layers = nn.ModuleList([
            GATLayer(nfeat, nhid_gat, dropout, alpha, concat=True)
            for _ in range(nheads)
        ])
        self.out_att = GATLayer(nhid_gat * nheads, nhid_gat, dropout, alpha, concat=False)

        # BiLSTM layer
        self.lstm = nn.LSTM(nhid_gat, nhid_lstm, num_layers=2, bias=True, batch_first=False, dropout=0, bidirectional=True)

        # Linear layers for final prediction
        self.active_index = nn.Linear(nhid_lstm * 2, 1)
        self.consume_index = nn.Linear(nhid_lstm * 2, 1)

    def forward(self, x_date, x_feature, x_mask_data):
        x_gat = x_feature
        print(x_gat.shape)
        # GAT layers
        x_gat = torch.cat([head(x_gat, x_mask_data) for head in self.gat_layers], dim=-1)  # (N, nheads*nhid)
        x_gat = F.dropout(x_gat, self.dropout, training=self.training)
        print(x_gat.shape)
        x_gat = self.out_att(x_gat, x_mask_data)
        print(x_gat.shape)

        # BiLSTM layer
        lstm_out, (hidden, cell) = self.lstm(x_gat)
        print(lstm_out.shape)
        x_lstm = F.dropout(lstm_out, self.dropout, training=self.training)

        # Final prediction
        act_pre = self.active_index(x_lstm)
        con_pre = self.consume_index(x_lstm)
        print(act_pre.shape,con_pre.shape)

        return act_pre, con_pre

