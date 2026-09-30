import numpy as np
import pandas as pd
import scipy.sparse
import scipy.io
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser('Generate matrices')
    parser.add_argument('--cells', type=int, help='Number of cells in matrix', required=True)
    parser.add_argument('--genes', type=int, help='Number of genes in matrix', required=True)
    parser.add_argument('--sparsity', type=float, help='Sparsity of matrix', required=True)
    parser.add_argument('--output', help='Output path', required=True)

    return parser.parse_args()


def main(args):
    matrix = scipy.sparse.random(args.cells, args.genes, density=1.0 - args.sparsity, format='csr', random_state=42)

    scipy.io.mmwrite(args.output, matrix)


if __name__ == '__main__':
    args = parse_args()
    main(args)
