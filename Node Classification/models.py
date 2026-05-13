import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_scatter import scatter

class EHGNN_MLP(torch.nn.Module):
    """多层感知机（MLP），可选输出 L2 归一化。"""
    def __init__(self, in_feat, hidden, out_feat, n_layer, wo_l2, dropout):
        super(EHGNN_MLP, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_feat, hidden))
        for _ in range(n_layer - 2):
            self.lins.append(torch.nn.Linear(hidden, hidden))
        self.lins.append(torch.nn.Linear(hidden, out_feat))
        self.wo_l2 = wo_l2
        self.dropout = dropout
        self.epsilon = torch.FloatTensor([1e-12])

    def l2_norm(self, x, device):
        # 与 TensorFlow 的 tf.l2_normalize 等价：逐行 L2 归一化，防止除零见 epsilon
        return x / (torch.max(torch.norm(x, dim=1, keepdim=True), self.epsilon.to(device)))

    def forward(self, x, device):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        if self.wo_l2:
            return x
        else:
            return self.l2_norm(x, device)

class EHGNN(torch.nn.Module):
    """EHGNN（Efficient HGNN）：meta-path 邻居聚合 + 可学习 MWeight/TWeight + α 融合自身表征。"""
    def __init__(self, in_feat, hidden, out_feat, n_layer, alpha, n_metapath, n_types, wo_l2, wo_mweight, wo_tweight, dropout):
        super(EHGNN, self).__init__()
        self.out_feat = out_feat
        self.alpha = alpha
        self.metapath_weights = torch.nn.Parameter((torch.zeros(n_metapath) + 1.))
        self.ntype_weights = torch.nn.Parameter((torch.zeros(n_types) + 1.))
        self.wo_mweight = wo_mweight
        self.wo_tweight = wo_tweight
        self.mlp = EHGNN_MLP(in_feat, hidden, out_feat, n_layer, wo_l2, dropout)

    def forward(self, X, s_features, s_idxs, t_idxs, weightss, t_typess, batch_size, device):
        # 调试时可打印 self.ntype_weights
        out = torch.zeros((batch_size, self.out_feat)).to(device)

        if self.wo_mweight:
            nm_weigths = torch.ones(self.metapath_weights.shape[0], device=device)
        else:
            nm_weigths = torch.softmax(self.metapath_weights, dim=0)
        if self.wo_tweight:
            nt_weights = torch.ones(self.ntype_weights.shape[0], device=device)
        else:
            nt_weights = torch.softmax(self.ntype_weights, dim=0)

        for i in range(len(s_idxs)):
            s_idx = s_idxs[i]
            t_idx = t_idxs[i]
            weights = weightss[i]
            t_types = t_typess[i]
            m_out = torch.zeros((batch_size, self.out_feat)).to(device)
            for t_type in t_types:
                type_out = self.mlp(X[t_type][t_idx[t_type]].to(device), device)
                type_out = scatter(type_out * weights[t_type][:, None].to(device), s_idx[t_type][:, None].to(device), dim=0, dim_size=batch_size, reduce='sum')
                m_out = m_out + type_out * nt_weights[t_type]
            out = out + m_out * nm_weigths[i]
        out = (1 - self.alpha) * out + self.alpha * self.mlp(s_features.to(device), device)
        return out