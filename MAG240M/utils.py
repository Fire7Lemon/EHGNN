import numpy as np
import os
import scipy.sparse as sp
import torch
import dgl
import time
from collections import Counter
from ogb.lsc import MAG240MDataset
from sklearn.metrics import f1_score
import random

def load_240m(graph_path):
    """从 OGB MAG240MDataset 加载异质图与论文节点标签、train/valid 划分。"""
    graph_path = os.path.abspath(os.path.expanduser(graph_path))
    print("Load Data")
    dataset = MAG240MDataset(root=graph_path)
    ei_writes = dataset.edge_index('author', 'writes', 'paper')
    ei_cites = dataset.edge_index('paper', 'paper')
    ei_affiliated = dataset.edge_index('author', 'institution')
    labels = torch.LongTensor(dataset.paper_label)

    g = dgl.heterograph({
        ('paper', 'cites', 'paper'): (ei_cites[0], ei_cites[1]),
        ('author', 'writes', 'paper'): (ei_writes[0], ei_writes[1]),
        ('author', 'affiliated_with', 'institution'): (ei_affiliated[0], ei_affiliated[1]),
        ('paper', 'cites_r', 'paper'): (ei_cites[1], ei_cites[0]),
        ('paper', 'writes_r', 'author'): (ei_writes[1], ei_writes[0]),
        ('institution', 'affiliated_with_r', 'author'): (ei_affiliated[1], ei_affiliated[0])
    })

    ei_writes = None
    ei_cites = None
    ei_affiliated = None

    split_dict = dataset.get_idx_split()
    train_nid = split_dict['train']  # 训练论文节点索引（numpy）
    valid_nid = split_dict['valid']  # 验证论文节点索引（numpy）
    # split_dict['test'] 官方未必提供，此处未使用
    print("Train, Valid Number", len(train_nid) + len(valid_nid))

    ## 为不同关系类型生成稀疏邻接（此处保留变量准备后续扩展）
    paper_num = dataset.num_papers
    author_num = dataset.num_authors
    institution_num = dataset.num_institutions
    print("Nodes number : ", paper_num + author_num + institution_num)

    idx_train = torch.LongTensor(train_nid)
    idx_val = torch.LongTensor(valid_nid)

    return g, labels.unsqueeze(1), idx_train, idx_val

def get_need_nodes(matrixs, ttypes_num, ttypess):
    """从各 meta-path 稀疏矩阵中收集涉及到的作者 / 机构 / 论文全局 id。"""
    related_nodes = []
    for i in range(ttypes_num):
        related_nodes.append([])

    for i in range(len(ttypess)):
        types = ttypess[i]
        matrix = matrixs[i]
        for type in types:
            # 调试：print(matrix[type])
            need_node = list(matrix[type].indices)
            related_nodes[type].extend(need_node)

    for i in range(ttypes_num):
        related_nodes[i] = set(related_nodes[i])

    return related_nodes

def load_features(related_nodes, paper_path, other_path, is_normalize):
    """mmap 读取大规模特征，随机投影扩展维度，并构建全局 id→batch 内行号的 features_map。"""
    author_num = 122383112
    institution_num = 25721
    paper_num = 121751666
    author_file = 'author.npy'
    institution_file = 'institution.npy'
    paper_file = 'node_feat.npy'

    ## features / features_map 下标：0 author，1 institution，2 paper
    features = {}
    features_map = {}
    features_map[0] = torch.zeros(author_num, dtype=torch.long) - 1
    features_map[1] = torch.zeros(institution_num, dtype=torch.long) - 1
    features_map[2] = torch.zeros(paper_num, dtype=torch.long) - 1

    for i in range(len(related_nodes)):
        related_node = np.array(related_nodes[i])
        features[i] = torch.FloatTensor(len(related_node), 768).half()
        if i == 0 or i == 1:
            if i == 0:  # 作者特征 mmap + 随机线性扩展到 768 维
                print('Prepare Author')
                feature_temp = np.memmap(filename=other_path + author_file,
                                         mode='r',
                                         dtype=np.float16,
                                         shape=(author_num, 128))
            elif i == 1:  # 机构特征
                print('Prepare Institution')
                feature_temp = np.memmap(filename=other_path + institution_file,
                                         mode='r',
                                         dtype=np.float16,
                                         shape=(institution_num, 128))

            rand_weight = torch.Tensor(128, 768).uniform_(-0.5, 0.5)
            feature_temp = torch.matmul(torch.FloatTensor(feature_temp[related_node]), rand_weight)
            features[i] = feature_temp.half()
        elif i == 2:  # 论文 RoBERTa 768 维特征
            print('Prepare Paper')
            feature_temp = np.memmap(filename=paper_path + paper_file,
                                     mode='r',
                                     dtype=np.float16,
                                     shape=(paper_num, 768))
            features[i] = torch.FloatTensor(feature_temp[related_node]).half()

        features_map[i][related_node] = torch.LongTensor([_ for _ in range(len(related_node))])

        if is_normalize:
            features[i] = features[i] / torch.sum(features[i], dim=1, keepdim=True)

    return features, features_map

def random_walk_sim(batch_idx, g, metapath, num_per_node, K, random_flag):
    """与 NC 类似，但 K 可为 list：按类型分别截断 Top-K 邻居。"""
    if torch.is_tensor(batch_idx):
        nodes_list = batch_idx.detach().cpu().numpy().astype(np.int64).ravel().tolist()
    else:
        nodes_list = [int(x) for x in list(batch_idx)]
    nodes_list = [int(v) for v in nodes_list for _ in range(num_per_node)]
    walks, types = dgl.sampling.random_walk(g=g, nodes=nodes_list, metapath=metapath)
    s_type = types[0]
    num_s = g.num_nodes(g.ntypes[s_type])
    tnode_types = set(types[1:].tolist())  # 去掉游走起点类型
    row_nodes = {}
    col_nodes = {}
    topk_counts = {}
    sim_mitrix = {}

    for t_type in tnode_types:
        row_nodes[t_type] = []
        col_nodes[t_type] = []
        topk_counts[t_type] = []

    if random_flag:
        for i in range(batch_idx.shape[0]):
            s_node = int(walks[i * num_per_node, 0])
            for t_type in tnode_types:
                t_indexs = torch.nonzero(types == t_type)

                if t_type == types[0]:
                    # 同源类型时跳过序列首位置，避免源点作邻居
                    t_indexs = t_indexs[1:]

                t_nodes = []
                for t_index in t_indexs:
                    t_node = walks[i * num_per_node:(i + 1) * num_per_node, t_index].squeeze().tolist()
                    t_nodes.extend(t_node)

                t_nodes = list(filter(lambda x: x != -1, t_nodes))
                if len(t_nodes) != 0:
                    k_cap = K[t_type] if isinstance(K, (list, tuple)) else K
                    if len(t_nodes) >= k_cap:
                        topk_node = random.sample(t_nodes, k_cap)
                    else:
                        topk_node = t_nodes
                    topk_count = [1 for _ in topk_node]
                    row_nodes[t_type].extend([s_node for _ in range(len(topk_node))])
                    col_nodes[t_type].extend(topk_node)
                    topk_counts[t_type].extend(topk_count / np.sum(topk_count))
    else:
        for i in range(batch_idx.shape[0]):
            s_node = int(walks[i * num_per_node, 0])
            for t_type in tnode_types:
                t_indexs = torch.nonzero(types == t_type)

                if t_type == types[0]:
                    # 同源类型时跳过序列首位置，避免源点作邻居
                    t_indexs = t_indexs[1:]

                t_nodes = []
                for t_index in t_indexs:
                    t_node = walks[i * num_per_node:(i + 1) * num_per_node, t_index].squeeze().tolist()
                    t_nodes.extend(t_node)

                t_nodes = list(filter(lambda x: x != -1, t_nodes))
                if len(t_nodes) != 0:
                    topk = Counter(t_nodes).most_common(K[t_type])
                    topk_node = [_[0] for _ in topk]
                    topk_count = [_[1] for _ in topk]
                    row_nodes[t_type].extend([s_node for _ in range(len(topk_node))])
                    col_nodes[t_type].extend(topk_node)
                    topk_counts[t_type].extend(topk_count / np.sum(topk_count))

    for t_type in tnode_types:
        num_t = g.num_nodes(g.ntypes[t_type])
        sim_mitrix[t_type] = sp.csr_matrix((topk_counts[t_type], (row_nodes[t_type], col_nodes[t_type])), shape=(num_s, num_t))

    return sim_mitrix, list(tnode_types), int(s_type)

def get_weights_sidx(sim_matrix, idx):
    if torch.is_tensor(idx):
        idx = idx.detach().cpu().numpy().astype(np.int64, copy=False).ravel()
    sim_matrix = sim_matrix[idx]
    s_idx, t_idx = sim_matrix.nonzero()
    s_idx = torch.LongTensor(s_idx)
    weights = torch.FloatTensor(sim_matrix.data)
    return s_idx, t_idx, weights

def get_model_need(n_metapaths, sim_matrixs, t_typess, batch):
    s_idxs = []
    t_idxs = []
    weightss = []
    for i in range(n_metapaths):
        sim_matrix = sim_matrixs[i]
        t_types = t_typess[i]
        s_idx = {}
        t_idx = {}
        weights = {}
        for t_type in t_types:
            s_idx[t_type], t_idx[t_type], weights[t_type] = get_weights_sidx(sim_matrix[t_type], batch)
        s_idxs.append(s_idx)
        t_idxs.append(t_idx)
        weightss.append(weights)
    return  s_idxs, t_idxs, weightss

def accuracy(output, labels):
    preds = output.max(dim=1)[1].view(-1, 1).cpu()
    correct = labels.cpu()
    macro_f1 = f1_score(correct, preds, average='macro')
    micro_f1 = f1_score(correct, preds, average='micro')
    return macro_f1, micro_f1