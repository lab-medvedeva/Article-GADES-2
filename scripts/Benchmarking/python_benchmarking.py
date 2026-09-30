import os
import gc
os.environ['OPENBLAS_NUM_THREADS'] = '24'

from cuml.metrics.pairwise_distances import sparse_pairwise_distances, pairwise_distances
from scipy.stats import kendalltau
from scipy.spatial.distance import pdist
#from sklearn.metrics import pairwise_distances
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import time
from tqdm import tqdm
from multiprocessing import Pool
from argparse import ArgumentParser
from psutil import Process
import json
from tqdm import tqdm
import scipy.io
from tqdm.contrib.concurrent import process_map
import psutil
import sys
import gc

def parse_args():
    parser = ArgumentParser('Python benchmarking')
    parser.add_argument('--num_threads', default=24, type=int)
    parser.add_argument('--metric', required=True, choices=['euclidean', 'pearson', 'kendall', 'cosine', 'spearman', 'manhattan'])
    parser.add_argument('--method', required=True, choices=['pandas', 'pythonic', 'raft'])
    parser.add_argument('--input', required=True, help='Path to dataset')
    parser.add_argument('--times', required=True, help='How many times to do benchmarking', type=int)
    parser.add_argument('--output', required=True, help='Path to output file')    
    parser.add_argument('--sparse', action='store_true', help='Whether to run code in GPU')
    parser.add_argument('--no-transpose', action='store_true', dest='no_transpose',
                        help='Do not transpose MTX data. Use for datasets where rows=samples (Generated, Recommendation)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    do_transpose = not args.no_transpose
    if args.input.endswith('.mtx'):
        mtx = scipy.io.mmread(args.input)

        if not args.sparse:
            src = mtx.T if do_transpose else mtx
            df = pd.DataFrame.sparse.from_spmatrix(src)
            np_array = df.values.astype(np.float32)

            delta = sys.getsizeof(np_array)
        else:
            mtx = mtx.tocsr()
            if do_transpose:
                mtx = mtx.T.tocsr()
            delta = mtx.data.nbytes + mtx.indices.nbytes + mtx.indptr.nbytes
    else:
        # CSV files always have rows=samples, cols=features — no transpose needed
        df = pd.read_csv(args.input, index_col=0)
        np_array = df.values.astype(np.float32)
        delta = sys.getsizeof(np_array) 
    if not args.sparse:

        print(np_array.shape)
    else:
        print(mtx.shape)

    def calculate_euclidean(indices):
        i, j = indices
        delta = np_array[i] - np_array[j]
        return np.sum(delta * delta)

    def calculate_pearson(indices):
        i, j = indices
        return (1.0 - np.corrcoef(np_array[i], np_array[j])[0, 1]) / 2.0
 
    def calculate_cosine(indices):
        i, j = indices
        return (1.0 - np.corrcoef(np_array[i], np_array[j])[0, 1]) / 2.0
 
    def calculate_spearman(indices):
        i, j = indices
        return (1.0 - spearmanr(np_array[i], np_array[j]).correlation) / 2.0
    
    def calculate_kendall(indices):
        i, j = indices
        return (1.0 - kendalltau(np_array[i], np_array[j]).correlation) / 2.0
    
    def calculate_manhattan(indices):
        i, j = indices
        return np.sum(np.abs(np_array[i] - np_array[j]))

    def kendall(a, b):
        return (1.0 - kendalltau(a, b).correlation) / 2.0

    def euclidean(a, b):
        delta = a - b
        return np.sum(delta * delta)
   
    def cosine(a, b):
        return (1.0 - np.corrcoef(a, b)[0, 1]) / 2.0

    def manhattan(a, b):
        return np.sum(np.abs(a - b))

    times = []
    
    functions = {
        'kendall': calculate_kendall,
        'pearson': calculate_pearson,
        'euclidean': calculate_euclidean,
        'cosine': calculate_cosine,
        'spearman': calculate_spearman,
        'manhattan': calculate_manhattan,
    }
    
    function = functions[args.metric]
    process = Process()
    memories = []
    for iteration_index in tqdm(range(args.times)):
        base_memory_usage = process.memory_info().rss
        start = time.time()
        if args.method == 'pythonic':
            
            if args.metric == 'euclidean':
                list_indices = [(i, j) for i in range(0, np_array.shape[0]) for j in range(i, np_array.shape[0])]


                output = np.zeros((np_array.shape[0], np_array.shape[0]), dtype=np.float32)

                #with Pool(args.num_threads) as p:
                output_results = process_map(function, list_indices, chunksize=100)
                for (i, j), distance in zip(list_indices, output_results):
                    output[i, j] = distance
                    output[j, i] = distance

                #output = pairwise_distances(np_array, metric='euclidean', n_jobs=24)
            elif args.metric == 'pearson':
                output = (1.0 - np.corrcoef(np_array)) / 2.0
            else:
                #output = pairwise_distances(np_array, metric=kendall, n_jobs=24)
                list_indices = [(i, j) for i in range(0, np_array.shape[0]) for j in range(i, np_array.shape[0])]


                output = np.zeros((np_array.shape[0], np_array.shape[0]), dtype=np.float32)

                #with Pool(args.num_threads) as p:
                output_results = process_map(function, list_indices, chunksize=10000)
                for (i, j), distance in zip(list_indices, output_results):
                    output[i, j] = distance
                    output[j, i] = distance
                
        elif args.method == 'pandas':
            if args.metric == 'euclidean':
                metric = euclidean
            elif args.metric == 'cosine':
                metric = cosine
            elif args.metric == 'manhattan':
                metric = manhattan
            else:
                metric = args.metric
            output = df.T.corr(method=metric)
        else:
            # raft / cuml.
            # cuml.pairwise_distances does NOT support 'pearson' directly, but
            # Pearson distance = 1 - corr(a,b) = cosine(center(a), center(b)),
            # so we center rows (per-sample mean subtraction) and dispatch to
            # cosine. Sparse pearson requires densification (centering destroys
            # sparsity), so we densify here as well — expect OOM on very large
            # sparse inputs, which the orchestrator will ban.
            if args.metric == 'pearson':
                if args.sparse:
                    arr = mtx.astype(np.float32).toarray()
                else:
                    arr = np_array.astype(np.float32)
                arr = arr - arr.mean(axis=1, keepdims=True)
                output = pairwise_distances(arr, metric='cosine')
            else:
                if args.sparse:
                    output = sparse_pairwise_distances(mtx.astype(np.float32), metric=args.metric)
                else:
                    output = pairwise_distances(np_array.astype(np.float32), metric=args.metric)
            # ROUND-TRIP: force the result back to host RAM inside the timed
            # region (same scope GADES/armadillo measure: host data -> host
            # matrix). cuML's default output_type follows the input, so numpy
            # input already returns numpy (D2H done); .get() handles the case
            # where a cupy/device array is returned instead. No-op for numpy.
            if hasattr(output, 'get'):
                output = output.get()

        result_memory_usage = process.memory_info().rss
        output_usage = sys.getsizeof(output)

        memory_usage = result_memory_usage - base_memory_usage - output_usage + delta

        memories.append({
            'base': base_memory_usage,
            'result': result_memory_usage,
            'output': output_usage,
            'found': memory_usage,
            'delta': delta,
        })

        print(output.shape)
        end = time.time()

        diff = (end - start) * 1000000

        #if args.method == 'pythonic':
        #    diff *= args.num_threads
        times.append(diff)
    
        result = pd.DataFrame(times)
        result.to_csv(args.output, index=None)
        result_memories = pd.DataFrame(memories)
        result_memories.to_csv(args.output.replace('.csv', '_memory.csv'), index=None)

        gc.collect()

    print(np.mean(times), np.std(times), np.max(times),times)
    data = {'name':args.input,'method':args.method,'metric':args.metric,'mean':np.mean(times),'std':np.std(times),'values':times,'max':np.max(times),'min':np.min(times)}
    json.dump(data,open('py_bench.json','a'))
