#!/usr/bin/env python3
"""Convert the CSR matrix X of an h5ad file to Matrix Market, objects by features.

    python3 h5ad_to_mtx.py <input.h5ad> <output.mtx>
"""
import sys

import h5py
import numpy as np

ROWS_PER_CHUNK = 50000


def main():
    source = sys.argv[1]
    target = sys.argv[2]
    with h5py.File(source, "r") as h5ad:
        matrix = h5ad["X"]
        encoding = matrix.attrs["encoding-type"]
        if encoding != "csr_matrix":
            raise SystemExit(f"X is {encoding}, expected csr_matrix")
        rows, columns = (int(size) for size in matrix.attrs["shape"])
        indptr = matrix["indptr"][:]
        total = int(indptr[-1])
        with open(target, "w") as mtx:
            mtx.write("%%MatrixMarket matrix coordinate real general\n")
            mtx.write(f"{rows} {columns} {total}\n")
            for first in range(0, rows, ROWS_PER_CHUNK):
                last = min(first + ROWS_PER_CHUNK, rows)
                begin = int(indptr[first])
                end = int(indptr[last])
                counts = np.diff(indptr[first:last + 1])
                row_numbers = np.repeat(np.arange(first + 1, last + 1), counts)
                column_numbers = matrix["indices"][begin:end].astype(np.int64) + 1
                values = matrix["data"][begin:end]
                block = np.column_stack((row_numbers, column_numbers, values))
                np.savetxt(mtx, block, fmt="%d %d %.9g")
    print(f"{target}: {rows} x {columns}, {total} non-zeros")


if __name__ == "__main__":
    main()
