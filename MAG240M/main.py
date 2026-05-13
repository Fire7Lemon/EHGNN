import argparse
import os
import torch
import time
import warnings
import numpy as np
from utils import random_walk_sim, accuracy, get_model_need, load_240m, get_need_nodes, load_features
from models import EHGNN

warnings.filterwarnings('ignore')


def _ensure_sep(path):
    """load_features 会把路径与文件名拼接；保证目录路径以分隔符结尾。"""
    path = os.path.abspath(os.path.expanduser(path))
    if not path.endswith(os.sep):
        path = path + os.sep
    return path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='../data/',
                        help='传给 ogb.lsc.MAG240MDataset(root=...) 的父目录（下载后出现 mag240m_kddcup2021/）')
    parser.add_argument('--data_root', type=str, default=None,
                        help='与 --path 同义别名（Linux 友好）；若指定则覆盖 --path')
    parser.add_argument('--feature_path', type=str, default=None,
                        help='论文特征目录（含 node_feat.npy）；默认 <root>/mag240m_kddcup2021/processed/paper/')
    parser.add_argument('--other_feature_path', type=str, default=None,
                        help='作者/机构等辅助特征目录；默认 <root>/mag240m_kddcup2021/')
    parser.add_argument('--is_normalize', action='store_true', help='特征按行归一化')
    parser.add_argument('--wo_l2', action='store_true', help='关闭 MLP 输出 L2 归一化')
    parser.add_argument('--wo_mweight', action='store_true', help='关闭 meta-path 权重')
    parser.add_argument('--wo_tweight', action='store_true', help='关闭节点类型权重')
    parser.add_argument('--r_neighbor', action='store_true', help='RW 后随机邻居 Top-K')
    parser.add_argument('--K_a', type=int, default=10, help='作者邻居 Top-K')
    parser.add_argument('--K_i', type=int, default=5, help='机构邻居 Top-K')
    parser.add_argument('--K_p', type=int, default=20, help='论文邻居 Top-K')
    parser.add_argument('--walk_num', type=int, default=40, help='每节点 RW 次数')
    parser.add_argument('--hidden', type=int, default=256, help='MLP 隐藏维度')
    parser.add_argument('--n_layers', type=int, default=4, help='MLP 层数')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout')
    parser.add_argument('--alpha', type=float, default=0.5, help='融合系数 α（自身与邻居）')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--val_epochs', type=int, default=5, help='验证间隔 epoch')
    parser.add_argument('--lr', type=float, default=0.003, help='学习率')
    parser.add_argument('--batch_size', type=int, default=3000, help='批大小')
    parser.add_argument('--gpu', type=int, default=0, help='GPU 编号')
    return parser.parse_args()

metapaths = []
# 关系语义涉及 cites / writes / affiliated_with 等，下列为 meta-path 序列
metapaths.append(['writes_r', 'writes', 'writes_r', 'writes'])
metapaths.append(['cites', 'cites', 'cites', 'cites'])
metapaths.append(['cites_r', 'cites', 'cites_r', 'cites'])
metapaths.append(['writes_r', 'affiliated_with', 'affiliated_with_r', 'writes'])

if __name__ == '__main__':
    args = parse_args()
    graph_root = os.path.abspath(os.path.expanduser(args.data_root if args.data_root else args.path))
    mag_home = os.path.join(graph_root, 'mag240m_kddcup2021')
    if args.feature_path is None:
        feature_path = _ensure_sep(os.path.join(mag_home, 'processed', 'paper'))
    else:
        feature_path = _ensure_sep(args.feature_path)
    if args.other_feature_path is None:
        other_feature_path = _ensure_sep(mag_home)
    else:
        other_feature_path = _ensure_sep(args.other_feature_path)

    args.path = graph_root
    args.feature_path = feature_path
    args.other_feature_path = other_feature_path
    print(args)
    print('Resolved OGB root (parent of mag240m_kddcup2021, absolute):', graph_root)
    print('Expected dataset_dir:', mag_home, '| exists:', os.path.isdir(mag_home))
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu')

    print('My similarity.')
    print(metapaths)
    start = time.perf_counter()
    g, labels, idx_train, idx_test = load_240m(graph_root)
    end = time.perf_counter()
    print('Done Load Graph, Running time: {:.4f} Seconds'.format(end - start))

    top_k = [args.K_a, args.K_i, args.K_p]

    start = time.perf_counter()
    train_matrixs = []
    test_matrixs = []
    t_typess = []
    for metapath in metapaths:
        train_matrix, t_types, s_type = random_walk_sim(idx_train, g, metapath, args.walk_num, top_k, args.r_neighbor)
        test_matrix, _, _ = random_walk_sim(idx_test, g, metapath, args.walk_num, top_k, args.r_neighbor)

        train_matrixs.append(train_matrix)
        test_matrixs.append(test_matrix)
        t_typess.append(t_types)
    end = time.perf_counter()
    print('Done sim, Running time: {:.4f} Seconds'.format(end - start))

    # 论文 / 作者 / 机构三类节点上的邻居集合（set）
    train_related_nodes = get_need_nodes(train_matrixs, 3, t_typess)
    test_related_nodes = get_need_nodes(test_matrixs, 3, t_typess)
    all_related_nodes = []
    related_num = 0
    for i in range(3):
        all_related_nodes.append(list(train_related_nodes[i].union(test_related_nodes[i])))
        related_num += len(all_related_nodes[i])
    print('Related Nodes Num : ', related_num)

    print('Load Feature')
    start = time.perf_counter()
    features, features_map = load_features(all_related_nodes, feature_path, other_feature_path, args.is_normalize)
    end = time.perf_counter()
    print('Done Feature, Running time: {:.4f} Seconds'.format(end - start))

    out_dim = labels.max().item() + 1
    model = EHGNN(in_feat=features[0].shape[1],
                  hidden=args.hidden,
                  out_feat=out_dim,
                  n_layer=args.n_layers,
                  alpha=args.alpha,
                  n_metapath=len(metapaths),
                  n_types=len(g.ntypes),
                  wo_l2=args.wo_l2,
                  wo_mweight=args.wo_mweight,
                  wo_tweight=args.wo_tweight,
                  dropout=args.dropout,
                  ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fcn = torch.nn.NLLLoss()

    print('Begin Train.')
    for run in range(args.epochs):
        start = time.perf_counter()
        dataloader = torch.utils.data.DataLoader(idx_train, batch_size=args.batch_size, shuffle=True, drop_last=False)
        model.train()
        loss_avg = []
        ma_avg = []
        mi_avg =[]
        for batch in dataloader:
            optimizer.zero_grad()
            s_idxs, t_idxs, weightss = get_model_need(len(metapaths), train_matrixs, t_typess, batch)
            map_batch = features_map[s_type][batch]
            batch_out = model(features, features[s_type][map_batch], features_map, s_idxs, t_idxs, weightss, t_typess, batch.shape[0], device)
            y_true = labels[batch].to(device)
            loss = loss_fcn(batch_out, y_true.squeeze(dim=1))
            loss_avg.append(loss.item())
            macro_f1, micro_f1 = accuracy(batch_out, y_true)
            ma_avg.append(macro_f1)
            mi_avg.append(micro_f1)
            loss.backward()
            optimizer.step()

        end = time.perf_counter()
        print('Epoch : {}, loss : {:.4f}, macro f1 : {:.4f}, micro f1 : {:.4f}, Running time: {:.4f} Seconds'.
              format(run, np.array(loss_avg).mean(), np.array(ma_avg).mean(), np.array(mi_avg).mean(), end - start))

        if run % args.val_epochs == 0 and run != 0:
            with torch.no_grad():
                model.eval()

                # 验证集 / 测试推理
                start = time.perf_counter()
                test_loader = torch.utils.data.DataLoader(idx_test, batch_size=args.batch_size, shuffle=False, drop_last=False)
                test_out = torch.FloatTensor([])
                for batch_test in test_loader:
                    s_idxs, t_idxs, weightss = get_model_need(len(metapaths), test_matrixs, t_typess, batch_test)
                    map_batch = features_map[s_type][batch_test]
                    result = model(features, features[s_type][map_batch], features_map, s_idxs, t_idxs, weightss, t_typess, batch_test.shape[0], device).to('cpu')
                    test_out = torch.cat((test_out, result), dim=0)

                y_true = labels[idx_test]
                macro_f1, micro_f1 = accuracy(test_out, y_true)
                end = time.perf_counter()
                print('Test macro f1 : {:.4f}, micro f1 : {:.4f}, Time : {:.4f}'.format(macro_f1, micro_f1, end-start))