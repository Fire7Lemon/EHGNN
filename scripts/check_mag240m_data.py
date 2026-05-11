#!/usr/bin/env python3
"""Verify OGB-LSC MAG240M dataset is readable under --root (no training)."""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(description='Sanity-check MAG240MDataset(root)')
    parser.add_argument(
        '--root',
        type=str,
        required=True,
        help='Parent directory passed to MAG240MDataset (contains mag240m_kddcup2021/)',
    )
    args = parser.parse_args()
    root = os.path.abspath(os.path.expanduser(args.root))

    try:
        from ogb.lsc import MAG240MDataset
    except ImportError as e:
        print('ERROR: ogb not installed:', e, file=sys.stderr)
        sys.exit(1)

    print('MAG240MDataset(root={!r})'.format(root))
    dataset = MAG240MDataset(root=root)

    num_papers = int(dataset.num_papers)
    num_authors = int(dataset.num_authors)
    num_institutions = int(dataset.num_institutions)

    labels = dataset.paper_label
    lab = np.asarray(labels)
    labeled = lab >= 0
    num_classes = int(lab[labeled].max()) + 1 if labeled.any() else 0

    split_dict = dataset.get_idx_split()

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


if __name__ == '__main__':
    main()
