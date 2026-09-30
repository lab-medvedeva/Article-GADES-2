import numpy as np
import pandas as pd
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser('Generate matrices')
    parser.add_argument('--cells', type=int, help='Number of cells in matrix', required=True)
    parser.add_argument('--genes', type=int, help='Number of genes in matrix', required=True)
    parser.add_argument('--output', help='Output path', required=True)

    return parser.parse_args()


def main(args):
    rng = np.random.default_rng(42)
    mtrx = rng.integers(low=0, high=9, size=(args.cells, args.genes))
    df = pd.DataFrame(mtrx)
    df.index.name = 'qa'

    df.to_csv(args.output)


if __name__ == '__main__':
    args = parse_args()
    main(args)

