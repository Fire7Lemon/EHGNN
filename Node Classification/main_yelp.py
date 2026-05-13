import argparse
import torch
import torch.nn.functional as F
import time
import numpy as np
from utils import random_walk_sim, accuracy, get_model_need, load_Yelp
from models import EHGNN

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Yelp', help='数据集名称，传给 load_Yelp 拼接路径')
    parser.add_argument('--path', type=str, default='../data/', help='数据根路径前缀')
    parser.add_argument('--other_path', type=str, default='../data/ogbn_mag/', help='其它特征路径（入口未使用）')
    parser.add_argument('--is_normalize', action='store_true', help='特征是否按行归一化')
    parser.add_argument('--wo_l2', action='store_true', help='关闭 MLP 输出 L2 归一化')
    parser.add_argument('--wo_mweight', action='store_true', help='关闭 meta-path 可学习权重')
    parser.add_argument('--wo_tweight', action='store_true', help='关闭节点类型可学习权重')
    parser.add_argument('--r_neighbor', action='store_true', help='RW 后随机取邻居 Top-K')
    parser.add_argument('--K', type=int, default=20, help='相似邻居数量上限')
    parser.add_argument('--walk_num', type=int, default=40, help='每节点 meta-path 随机游走次数')
    parser.add_argument('--hidden', type=int, default=256, help='MLP 隐藏维度')
    parser.add_argument('--n_layers', type=int, default=4, help='MLP 层数')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout 比率')
    parser.add_argument('--eps', type=float, default=1e-5, help='PPR 数值项（当前 RW 未用）')
    parser.add_argument('--alpha', type=float, default=0.5, help='自身分支与邻居聚合的融合系数 α')
    parser.add_argument('--num_threads', type=int, default=40, help='预留线程数（当前未用）')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--val_epochs', type=int, default=5, help='验证间隔（epoch）')
    parser.add_argument('--lr', type=float, default=0.0003, help='学习率')
    parser.add_argument('--batch_size', type=int, default=3000, help='批大小')
    parser.add_argument('--gpu', type=int, default=0, help='GPU 编号')
    return parser.parse_args()

metapaths_yelp = []
metapaths_yelp.append(['locatedin', 'locatedin_r'])
metapaths_yelp.append(['rate', 'rate_r'])
metapaths_yelp.append(['describedwith', 'context', 'describedwith_r'])

# Yelp 多标签节点分类入口（损失为 BCELoss）
if __name__ == '__main__':
    args = parse_args()
    print(args)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu')

    print('My similarity.')
    metapaths = metapaths_yelp
    print(metapaths)
    start = time.perf_counter()
    g, features, labels, idx_train, idx_test = load_Yelp(args.path, args.dataset, args.is_normalize)
    end = time.perf_counter()
    print('Done Load Data, Running time: {:.4f} Seconds'.format(end - start))

    start = time.perf_counter()
    train_matrixs = []
    val_matrixs = []
    test_matrixs = []
    t_typess = []
    for metapath in metapaths:
        train_matrix, t_types, s_type = random_walk_sim(idx_train, g, metapath, args.walk_num, args.K, args.r_neighbor)
        test_matrix, _, _ = random_walk_sim(idx_test, g, metapath, args.walk_num, args.K, args.r_neighbor)

        train_matrixs.append(train_matrix)
        test_matrixs.append(test_matrix)
        t_typess.append(t_types)
    end = time.perf_counter()
    print('Done sim, Running time: {:.4f} Seconds'.format(end - start))

    out_dim = labels.shape[1]
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
    loss_fcn = torch.nn.BCELoss()

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
            batch_out = model(features, features[s_type][batch], s_idxs, t_idxs, weightss, t_typess, batch.shape[0], device).sigmoid()
            y_true = labels[batch].to(device)
            # 调试时可打印 batch_out / y_true 形状
            loss = loss_fcn(batch_out, y_true.float())
            loss_avg.append(loss.item())
            macro_f1, micro_f1 = accuracy(batch_out, y_true, args.dataset)
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
                # 测试集评估
                start = time.perf_counter()
                test_loader = torch.utils.data.DataLoader(idx_test, batch_size=args.batch_size, shuffle=False, drop_last=False)
                test_out = torch.FloatTensor([])
                for batch_test in test_loader:
                    s_idxs, t_idxs, weightss = get_model_need(len(metapaths), test_matrixs, t_typess, batch_test)
                    result = model(features, features[s_type][batch_test], s_idxs, t_idxs, weightss, t_typess, batch_test.shape[0], device).to('cpu').sigmoid()
                    test_out = torch.cat((test_out, result), dim=0)

                y_true = labels[idx_test]
                macro_f1, micro_f1 = accuracy(test_out, y_true, args.dataset)
                end = time.perf_counter()
                print('Test macro f1 : {:.4f}, micro f1 : {:.4f}, Time : {:.4f}'.format(macro_f1, micro_f1, end-start))