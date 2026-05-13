import argparse
import os
import random
import torch
import torch.nn.functional as F
import time
import numpy as np

from utils import random_walk_sim, accuracy, get_model_need, load_dblp, load_PubMed
from models import EHGNN

# 命令行参数解析
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='PubMed', help='数据集名称（PubMed / DBLP），决定加载逻辑与 meta-path 列表')
    parser.add_argument('--path', type=str, default='../data/', help='数据根路径前缀，与数据集名拼接为 data_path+name/')
    parser.add_argument('--other_path', type=str, default='../data/ogbn_mag/', help='其它特征路径（当前入口未使用，仅占位）')
    parser.add_argument('--is_normalize', action='store_true', help='特征是否按行归一化（每行和为 1）')
    parser.add_argument('--wo_l2', action='store_true', help='关闭 MLP（Multi-Layer Perceptron，多层感知机）输出的 L2 归一化')
    parser.add_argument('--wo_mweight', action='store_true', help='关闭 meta-path 可学习权重（MWeight），改为均匀权重')
    parser.add_argument('--wo_tweight', action='store_true', help='关闭目标节点类型可学习权重（TWeight），改为均匀权重')
    parser.add_argument('--r_neighbor', action='store_true', help='RW 后对邻居随机采样 Top-K，而非按出现频次')
    parser.add_argument('--neighbor_strategy', type=str, default='freq',
                        choices=['freq', 'random', 'hybrid', 'temp'],
                        help='邻居选择：freq=频次 Top-K（默认）；random/hybrid/temp 见 select_neighbors_by_strategy')
    parser.add_argument('--hybrid_ratio', type=float, default=0.8,
                        help='hybrid：高频槽占比 int(K*ratio)，其余槽从剩余唯一候选随机')
    parser.add_argument('--temp', type=float, default=1.0,
                        help='temp：p_i ∝ count_i^(1/temp)，无放回采样至多 K 个')
    parser.add_argument('--K', type=int, default=20, help='每个源节点保留的相似邻居数量上限（Top-K）')
    parser.add_argument('--walk_num', type=int, default=40, help='每个节点沿 meta-path 执行的随机游走次数')
    parser.add_argument('--hidden', type=int, default=256, help='MLP 隐藏层维度')
    parser.add_argument('--n_layers', type=int, default=4, help='MLP 层数')
    parser.add_argument('--dropout', type=float, default=0.4, help='Dropout 比率')
    parser.add_argument('--eps', type=float, default=1e-5, help='与 PPR（Personalized PageRank，个性化 PageRank）相关的数值稳定项（当前 RW 流程未使用）')
    parser.add_argument('--alpha', type=float, default=0.7, help='融合系数：自身 MLP 表征与邻居聚合表征的加权（见 EHGNN.forward）')
    parser.add_argument('--num_threads', type=int, default=40, help='预留：PPR / RW 线程数（当前 DGL random_walk 路径未使用）')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--val_epochs', type=int, default=5, help='每隔多少 epoch 在测试集上验证一次')
    parser.add_argument('--lr', type=float, default=1e-3, help='Adam 学习率')
    parser.add_argument('--batch_size', type=int, default=3000, help='训练批大小')
    parser.add_argument('--gpu', type=int, default=0, help='CUDA 设备编号')
    parser.add_argument('--seed', type=int, default=42, help='随机种子（random / numpy / torch）')
    return parser.parse_args()


metapaths_dblp = []
metapaths_dblp.append(['study', 'cooccur', 'study_r'])
metapaths_dblp.append(['coauthor', 'coauthor'])
metapaths_dblp.append(['cite', 'cite', 'cite'])
metapaths_dblp.append(['publishin', 'publishin_r'])
metapaths_dblp.append(['activein', 'activein_r'])

metapaths_pubmed = []
metapaths_pubmed.append(['dad_r', 'dad'])
metapaths_pubmed.append(['gcd_r', 'gcd'])
metapaths_pubmed.append(['gcd_r', 'gag', 'gcd'])
metapaths_pubmed.append(['cid_r', 'cid'])
metapaths_pubmed.append(['cid_r', 'cig', 'cig_r', 'cac', 'cis', 'cis_r', 'cid'])
metapaths_pubmed.append(['swd_r', 'sas', 'swd'])


if __name__ == '__main__':
    args = parse_args()
    if not (0.0 <= args.hybrid_ratio <= 1.0):
        raise ValueError('hybrid_ratio must be in [0, 1], got {}'.format(args.hybrid_ratio))
    if args.temp <= 0:
        raise ValueError('temp must be > 0, got {}'.format(args.temp))
    neighbor_effective = 'random' if args.r_neighbor else args.neighbor_strategy
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(args)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu')

    print('My similarity.')
    start = time.perf_counter()
    if args.dataset == 'DBLP':
        g, features, labels, idx_train, idx_test = load_dblp(args.path, args.dataset, args.is_normalize)
        metapaths = metapaths_dblp
    elif args.dataset == 'PubMed':
        g, features, labels, idx_train, idx_test = load_PubMed(args.path, args.dataset, args.is_normalize)
        metapaths = metapaths_pubmed
    end = time.perf_counter()
    print('Done Load Data, Running time: {:.4f} Seconds'.format(end - start))

    print(metapaths)

    start = time.perf_counter()
    train_matrixs = []
    test_matrixs = []
    t_typess = []
    for metapath in metapaths:
        train_matrix, t_types, s_type = random_walk_sim(
            idx_train, g, metapath, args.walk_num, args.K, args.r_neighbor,
            neighbor_strategy=args.neighbor_strategy, hybrid_ratio=args.hybrid_ratio, temp=args.temp)
        test_matrix, _, _ = random_walk_sim(
            idx_test, g, metapath, args.walk_num, args.K, args.r_neighbor,
            neighbor_strategy=args.neighbor_strategy, hybrid_ratio=args.hybrid_ratio, temp=args.temp)
        train_matrixs.append(train_matrix)
        test_matrixs.append(test_matrix)
        t_typess.append(t_types)
    end = time.perf_counter()
    print('Done sim, Running time: {:.4f} Seconds'.format(end - start))

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

    best_test_macro = -1.0
    best_test_micro = -1.0
    best_epoch = -1
    final_test_macro = None
    final_test_micro = None

    print('Begin Train.')
    train_begin = time.perf_counter()
    for run in range(args.epochs):
        start = time.perf_counter()
        dataloader = torch.utils.data.DataLoader(idx_train, batch_size=args.batch_size, shuffle=True, drop_last=False)
        model.train()
        loss_avg = []
        ma_avg = []
        mi_avg = []
        for batch in dataloader:
            optimizer.zero_grad()
            s_idxs, t_idxs, weightss = get_model_need(len(metapaths), train_matrixs, t_typess, batch)
            batch_out = model(features, features[s_type][batch], s_idxs, t_idxs, weightss, t_typess, batch.shape[0], device)
            batch_out = F.log_softmax(batch_out, dim=1)
            y_true = labels[batch].to(device)
            loss = loss_fcn(batch_out, y_true.squeeze(dim=1))
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

                start_ev = time.perf_counter()
                test_loader = torch.utils.data.DataLoader(idx_test, batch_size=args.batch_size, shuffle=False, drop_last=False)
                test_out = torch.FloatTensor([])
                for batch_test in test_loader:
                    s_idxs, t_idxs, weightss = get_model_need(len(metapaths), test_matrixs, t_typess, batch_test)
                    result = model(features, features[s_type][batch_test], s_idxs, t_idxs, weightss, t_typess, batch_test.shape[0], device).to('cpu')
                    result = F.log_softmax(result, dim=1)
                    test_out = torch.cat((test_out, result), dim=0)

                y_true = labels[idx_test]
                macro_f1, micro_f1 = accuracy(test_out, y_true, args.dataset)
                end_ev = time.perf_counter()
                final_test_macro, final_test_micro = macro_f1, micro_f1
                if macro_f1 > best_test_macro:
                    best_test_macro = macro_f1
                    best_test_micro = micro_f1
                    best_epoch = run
                print('Test macro f1 : {:.4f}, micro f1 : {:.4f}, Time : {:.4f}'.format(macro_f1, micro_f1, end_ev - start_ev))

    final_epoch = args.epochs - 1 if args.epochs > 0 else -1
    last_eval_at_final_epoch = (
        args.epochs > 0
        and final_epoch != 0
        and final_epoch % args.val_epochs == 0)
    if args.epochs > 0 and not last_eval_at_final_epoch:
        with torch.no_grad():
            model.eval()
            start_ev = time.perf_counter()
            test_loader = torch.utils.data.DataLoader(idx_test, batch_size=args.batch_size, shuffle=False, drop_last=False)
            test_out = torch.FloatTensor([])
            for batch_test in test_loader:
                s_idxs, t_idxs, weightss = get_model_need(len(metapaths), test_matrixs, t_typess, batch_test)
                result = model(features, features[s_type][batch_test], s_idxs, t_idxs, weightss, t_typess, batch_test.shape[0], device).to('cpu')
                result = F.log_softmax(result, dim=1)
                test_out = torch.cat((test_out, result), dim=0)
            y_true = labels[idx_test]
            macro_f1, micro_f1 = accuracy(test_out, y_true, args.dataset)
            end_ev = time.perf_counter()
            final_test_macro, final_test_micro = macro_f1, micro_f1
            print('Final epoch {} test macro f1 : {:.4f}, micro f1 : {:.4f}, Time : {:.4f}'.format(
                final_epoch, macro_f1, micro_f1, end_ev - start_ev))

    print('Best Test Macro-F1 : {:.4f}, Micro-F1 : {:.4f}, Epoch : {}'.format(
        best_test_macro, best_test_micro, best_epoch))
    print('Final Epoch : {}, Final Test Macro-F1 : {:.4f}, Micro-F1 : {:.4f}'.format(
        final_epoch, final_test_macro, final_test_micro))

    total_training_sec = time.perf_counter() - train_begin
    print('Total training time: {:.4f} s'.format(total_training_sec))

    if args.dataset == 'PubMed':
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
        os.makedirs(results_dir, exist_ok=True)
        out_path = os.path.join(results_dir, 'pubmed_nc_result.txt')
        lines = [
            'seed={}'.format(args.seed),
            'neighbor_strategy={}'.format(neighbor_effective),
            'hybrid_ratio={}'.format(args.hybrid_ratio),
            'temp={}'.format(args.temp),
            'best_test_macro={}'.format(best_test_macro),
            'best_test_micro={}'.format(best_test_micro),
            'best_epoch={}'.format(best_epoch),
            'final_epoch={}'.format(final_epoch),
            'final_test_macro={}'.format(final_test_macro),
            'final_test_micro={}'.format(final_test_micro),
            'total_training_time_sec={}'.format(total_training_sec),
            'alpha={} K={} lr={} dropout={} hidden={} layers={} batch_size={}'.format(
                args.alpha, args.K, args.lr, args.dropout, args.hidden, args.n_layers, args.batch_size),
        ]
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')
        print('Results saved to: {}'.format(out_path))
