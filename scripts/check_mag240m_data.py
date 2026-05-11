#!/usr/bin/env python3
"""Verify OGB-LSC MAG240M under --root.

WARNING: Calling MAG240MDataset(root=...) will trigger OGB download / preprocessing
if the dataset is missing or incomplete under <root>/mag240m_kddcup2021/.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np


def _exists_msg(path: str) -> str:
    return 'yes' if os.path.isfile(path) else 'no'


def _dir_exists_msg(path: str) -> str:
    return 'yes' if os.path.isdir(path) else 'no'


def main():
    parser = argparse.ArgumentParser(description='Sanity-check MAG240MDataset(root)')
    parser.add_argument(
        '--root',
        type=str,
        required=True,
        help='Parent directory for MAG240MDataset (contains mag240m_kddcup2021/); supports ~/ expansion',
    )
    args = parser.parse_args()
    root = os.path.abspath(os.path.expanduser(args.root))
    dataset_dir = os.path.join(root, 'mag240m_kddcup2021')

    print('resolved_root (absolute):', root)
    print('dataset_dir:', dataset_dir)
    print('dataset_dir exists (before MAG240MDataset):', _dir_exists_msg(dataset_dir))

    paper_feat = os.path.join(dataset_dir, 'processed', 'paper', 'node_feat.npy')
    print('paper_feat path:', paper_feat)
    print('paper_feat exists (before MAG240MDataset):', _exists_msg(paper_feat))

    print('')
    print('NOTE: The next step constructs MAG240MDataset(root).')
    print('      If data are missing, OGB may DOWNLOAD / PROCESS large files (hours to ~1 day).')
    print('')

    try:
        from ogb.lsc import MAG240MDataset
    except ImportError as e:
        print('ERROR: ogb not installed:', e, file=sys.stderr)
        sys.exit(1)

    dataset = MAG240MDataset(root=root)

    num_papers = int(dataset.num_papers)
    num_authors = int(dataset.num_authors)
    num_institutions = int(dataset.num_institutions)

    labels = dataset.paper_label
    lab = np.asarray(labels)
    labeled = lab >= 0
    num_classes = int(lab[labeled].max()) + 1 if labeled.any() else 0

    split_dict = dataset.get_idx_split()

    print('')
    print('--- after MAG240MDataset ---')
    print('dataset_dir exists:', _dir_exists_msg(dataset_dir))
    print('num_papers:', num_papers)
    print('num_authors:', num_authors)
    print('num_institutions:', num_institutions)
    print('num_classes (from labeled paper_label max+1):', num_classes)

    print('split sizes:')
    for name in ('train', 'valid', 'test'):
        if name in split_dict:
            print('  {:8s}: {}'.format(name, len(split_dict[name])))
        else:
            print('  {:8s}: (not in split_dict)'.format(name))
    print('split_dict keys:', list(split_dict.keys()))

    print('paper_label exists:', 'yes' if hasattr(dataset, 'paper_label') else 'no')
    if hasattr(dataset, 'paper_label'):
        pl = np.asarray(dataset.paper_label)
        print('paper_label shape:', pl.shape)

    print('paper_feat exists:', _exists_msg(paper_feat))

    year_ok = False
    year_detail = []
    for attr in ('paper_year', 'all_paper_year'):
        if hasattr(dataset, attr):
            year_ok = True
            val = getattr(dataset, attr)
            try:
                arr = np.asarray(val)
                year_detail.append('{} shape={}'.format(attr, arr.shape))
            except Exception:
                year_detail.append('{} present (non-array)'.format(attr))
    print('paper_year related:', 'yes' if year_ok else 'no')
    for line in year_detail:
        print('  ', line)


if __name__ == '__main__':
    main()
