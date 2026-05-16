import argparse
import random
import time

import numpy as np
import torch

from utils import random_walk_sim, get_model_need, accuracy, neg_sample, load_Yelp
from models import EHGNN_yelp


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Yelp', help='数据集名称')
    parser.add_argument('--path', type=str, default='../data/', help='数据根路径前缀')
    parser.add_argument('--is_normalize', action='store_true', help='特征按行归一化')
    parser.add_argument('--wo_l2', action='store_true', help='关闭 MLP L2 归一化')
    parser.add_argument('--wo_mweight', action='store_true', help='关闭 meta-path 权重')
    parser.add_argument('--wo_tweight', action='store_true', help='关闭类型权重')
    parser.add_argument('--r_neighbor', action='store_true', help='RW 随机邻居')
    parser.add_argument('--K', type=int, default=10, help='Top-K 邻居')
    parser.add_argument('--walk_num', type=int, default=100, help='RW 次数')
    parser.add_argument('--hidden', type=int, default=256, help='隐藏维度')
    parser.add_argument('--n_layers', type=int, default=2, help='MLP 层数')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout')
    parser.add_argument('--eps', type=float, default=1e-5, help='PPR eps（未用）')
    parser.add_argument('--alpha', type=float, default=0.5, help='融合系数 α')
    parser.add_argument('--num_threads', type=int, default=40, help='预留线程（未用）')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--val_epochs', type=int, default=5, help='验证间隔（步）')
    parser.add_argument('--lr', type=float, default=0.005, help='学习率')
    parser.add_argument('--batch_size', type=int, default=4000, help='批大小')
    parser.add_argument('--gpu', type=int, default=0, help='GPU 编号')
    parser.add_argument('--seed', type=int, default=42, help='random / numpy / torch 随机种子')
    return parser.parse_args()


metapaths_b = []
metapaths_b.append(['locatedin', 'locatedin_r'])
metapaths_b.append(['rate', 'rate_r'])
metapaths_b.append(['describedwith', 'context', 'describedwith_r'])

metapaths_p = []
metapaths_p.append(['context_r', 'context'])
metapaths_p.append(['context', 'context_r', 'context'])
metapaths_p.append(['describedwith_r', 'describedwith'])

## Yelp：business / phrase 两套 meta-path 相似矩阵
if __name__ == '__main__':
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(args)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu')

    print(metapaths_b)
    print(metapaths_p)
    start = time.perf_counter()
    g, features, train_links, test_links, labels = load_Yelp(args.path, args.dataset, args.is_normalize)

    num_business = 7474
    num_phrase = 74943

    end = time.perf_counter()
    print('Done Load Data, Running time: {:.4f} Seconds'.format(end - start))

    start = time.perf_counter()
    sim_matrixs_b = []
    t_typess_b = []
    for metapath in metapaths_b:
        sim_matrix, t_types, s_type_b = random_walk_sim(
            torch.LongTensor([_ for _ in range(num_business)]), g, metapath,
            args.walk_num, args.K, args.r_neighbor,
        )
        sim_matrixs_b.append(sim_matrix)
        t_typess_b.append(t_types)

    sim_matrixs_p = []
    t_typess_p = []
    for metapath in metapaths_p:
        sim_matrix, t_types, s_type_p = random_walk_sim(
            torch.LongTensor([_ for _ in range(num_phrase)]), g, metapath,
            args.walk_num, args.K, args.r_neighbor,
        )
        sim_matrixs_p.append(sim_matrix)
        t_typess_p.append(t_types)
    end = time.perf_counter()
    print('Done my sim, Running time: {:.4f} Seconds'.format(end - start))

    model = EHGNN_yelp(
        in_feat=features[0].shape[1],
        hidden=args.hidden,
        out_feat=args.hidden,
        n_layer=args.n_layers,
        alpha=args.alpha,
        n_metapath=len(metapaths_b) + len(metapaths_p),
        n_types=len(g.ntypes) * 2,
        wo_l2=args.wo_l2,
        wo_mweight=args.wo_mweight,
        wo_tweight=args.wo_tweight,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fcn = torch.nn.BCELoss()

    def evaluate_yelp_lp():
        """测试边 AUC 与 AP（训练日志里的 precision 即 sklearn average_precision_score）。"""
        with torch.no_grad():
            model.eval()
            test_loader = torch.utils.data.DataLoader(
                torch.LongTensor([_ for _ in range(test_links.shape[1])]),
                batch_size=5000,
                shuffle=False,
                drop_last=False,
            )
            test_out = torch.FloatTensor([]).to(device)
            for batch_test in test_loader:
                pos_s = test_links[0][batch_test]
                pos_t = test_links[1][batch_test]

                s_idxs, t_idxs, weightss = get_model_need(len(metapaths_b), sim_matrixs_b, t_typess_b, pos_s)
                pos_s_out = model(
                    features, features[s_type_b][pos_s], s_idxs, t_idxs, weightss,
                    t_typess_b, 0, pos_s.shape[0], device,
                )
                s_idxs, t_idxs, weightss = get_model_need(len(metapaths_p), sim_matrixs_p, t_typess_p, pos_t)
                pos_t_out = model(
                    features, features[s_type_p][pos_t], s_idxs, t_idxs, weightss,
                    t_typess_p, 1, pos_t.shape[0], device,
                )
                pos = (pos_s_out * pos_t_out).sum(dim=-1)

                batch_out = pos.sigmoid()
                test_out = torch.cat((test_out, batch_out), dim=0)

            y_true = labels
            auc, ap = accuracy(test_out.to('cpu'), y_true)
            return auc, ap

    best_test_auc = float('-inf')
    best_test_ap = float('-inf')
    best_epoch = -1

    print('Begin Train.')
    train_begin = time.perf_counter()
    for run in range(args.epochs):
        train_idx = torch.randint(0, train_links.shape[0], (num_phrase,))
        train_loader = torch.utils.data.DataLoader(train_idx, batch_size=args.batch_size, shuffle=True, drop_last=False)
        step = 0
        for batch_train in train_loader:
            model.train()
            step = step + 1
            start = time.perf_counter()
            optimizer.zero_grad()
            pos_s = train_links[0][batch_train]
            pos_t = train_links[1][batch_train]
            neg_s, neg_t = neg_sample(pos_s, 0, num_phrase)

            s_idxs, t_idxs, weightss = get_model_need(len(metapaths_b), sim_matrixs_b, t_typess_b, pos_s)
            pos_s_out = model(features, features[s_type_b][pos_s], s_idxs, t_idxs, weightss, t_typess_b, 0, pos_s.shape[0], device)
            s_idxs, t_idxs, weightss = get_model_need(len(metapaths_p), sim_matrixs_p, t_typess_p, pos_t)
            pos_t_out = model(features, features[s_type_p][pos_t], s_idxs, t_idxs, weightss, t_typess_p, 1, pos_t.shape[0], device)
            s_idxs, t_idxs, weightss = get_model_need(len(metapaths_b), sim_matrixs_b, t_typess_b, neg_s)
            neg_s_out = model(features, features[s_type_b][neg_s], s_idxs, t_idxs, weightss, t_typess_b, 0, neg_s.shape[0], device)
            s_idxs, t_idxs, weightss = get_model_need(len(metapaths_p), sim_matrixs_p, t_typess_p, neg_t)
            neg_t_out = model(features, features[s_type_p][neg_t], s_idxs, t_idxs, weightss, t_typess_p, 1, neg_t.shape[0], device)
            pos = (pos_s_out * pos_t_out).sum(dim=-1)
            neg = (neg_s_out * neg_t_out).sum(dim=-1)
            batch_out = torch.cat((pos, neg)).sigmoid()
            y_true = torch.cat((torch.ones(batch_train.shape[0], dtype=int), torch.zeros(batch_train.shape[0], dtype=int)))
            loss = loss_fcn(batch_out, y_true.float().to(device))
            auc, precision = accuracy(batch_out.to('cpu'), y_true.to('cpu'))
            loss.backward()
            optimizer.step()
            end = time.perf_counter()
            print(
                'Epoch : {}, Step : {}, loss : {:.4f}, auc : {:.4f}, precision : {:.4f}, Running time: {:.4f} Seconds'.
                format(run, step, loss.item(), auc, precision, end - start),
            )

            if step % args.val_epochs == 0 and step != 0:
                start_ev = time.perf_counter()
                auc, precision = evaluate_yelp_lp()
                end_ev = time.perf_counter()
                print(
                    'Test auc : {:.4f}, precision : {:.4f}, Time : {:.4f}'.format(
                        auc, precision, end_ev - start_ev,
                    ),
                )
                if auc > best_test_auc or (
                    auc == best_test_auc and precision > best_test_ap
                ):
                    best_test_auc = auc
                    best_test_ap = precision
                    best_epoch = run

    final_epoch = args.epochs - 1 if args.epochs > 0 else -1
    final_test_auc, final_test_ap = evaluate_yelp_lp()
    total_training_sec = time.perf_counter() - train_begin

    if best_epoch < 0:
        best_test_auc = final_test_auc
        best_test_ap = final_test_ap
        best_epoch = final_epoch

    print(
        'Best Test AUC : {:.4f}, AP : {:.4f}, Epoch : {}'.format(
            best_test_auc, best_test_ap, best_epoch,
        ),
    )
    print(
        'Final Epoch : {}, Final Test AUC : {:.4f}, AP : {:.4f}'.format(
            final_epoch, final_test_auc, final_test_ap,
        ),
    )
    print('Total training time: {:.4f} s'.format(total_training_sec))
