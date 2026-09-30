# Article-GADES-2

Benchmarks, results and figure scripts of the GADES 2.0 article. [GADES](https://github.com/lab-medvedeva/GADES-main) computes pairwise distance matrices on the CPU and on the GPU for dense and sparse inputs with six metrics: Euclidean, cosine, Pearson, Manhattan, Spearman and Kendall.

## Layout

```
results/GeneratedDense/     timings on generated dense matrices
results/GeneratedSparse/    timings on generated sparse matrices
results/RealDatasets/       timings on 22 real datasets
results/HugeDatasets/       timings on 10 huge datasets
scripts/MatricesGeneration/ generators of the dense and sparse matrices
scripts/Benchmarking/       benchmark drivers and the script that runs them
scripts/Validation/         accuracy checks against reference implementations
reproducibility/            scripts that collect the results and draw the figures
```

## Requirements

The article was measured on an HPC cluster with setup (12 cores, 24 threads) with 62 GB of RAM and an NVIDIA GeForce RTX 3090 (24 GB), Ubuntu 20.04, CUDA 12.1, R 4.1.3, Python 3.12.3 and cuML 26.02.

- GADES built from source, see below.
- R packages: `Matrix`, `glue`, `hdf5r`, `amap`, `factoextra`, `R.utils`.
- Python packages: `numpy`, `scipy`, `pandas`, `psutil`, `tqdm`, `h5py`, `cuml` (RAPIDS), and `matplotlib`, `seaborn` for the figures.
- [Armadillo](https://arma.sourceforge.net) and [ArrayFire](https://arrayfire.com) with the CUDA and CPU backends.

## Step 1. Build GADES and the baseline drivers

```shell
git clone https://github.com/lab-medvedeva/GADES-main.git
cd GADES-main
cmake -S . -B build -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=$PWD/build
cmake --build build
export GADES_ROOT=$PWD
```

The benchmark driver loads `$GADES_ROOT/build/mtrx.so`, `$GADES_ROOT/build/mtrx_cpu.so` and `$GADES_ROOT/R/mtrx.R`.

```shell
cd Article-GADES-2/scripts/Benchmarking
g++ -O3 -std=c++17 test_armadillo.cpp -o test_armadillo -larmadillo
g++ -O3 -std=c++17 test_arrayfire.cpp -o test_arrayfire -laf
```

For an ArrayFire build tree that is not installed system-wide, add `-I<arrayfire>/include -I<arrayfire>/build/include -L<arrayfire>/build/src/api/unified` and put the directories of its `unified`, `cuda` and `cpu` libraries into `LD_LIBRARY_PATH`.
## Docker

Instead of Step 1, the benchmark can run in a container. The image holds CUDA 12.1, R with GADES and the R baselines, Python 3.12 with cuML, pandas and SciPy, Armadillo, ArrayFire 3.10 and the compiled drivers. The host needs an NVIDIA driver of version 530 or newer, Docker and the NVIDIA Container Toolkit.

### Install the NVIDIA Container Toolkit

Once per host, on Ubuntu or Debian (see the [installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) for other systems):

```shell
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Check that containers see the GPU:

```shell
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

### Get the image

The image is published on Docker Hub:

```shell
docker pull akhtyamovpavel/article-gades-2
```

It can also be built from the root of this repository:

```shell
docker build -t akhtyamovpavel/article-gades-2 .
```

The build takes about an hour, most of it compiling ArrayFire, and the image takes about 16 GB. Build arguments:

| Argument | Default | Meaning |
|---|---|---|
| `GADES_COMMIT` | `main` | commit of https://github.com/lab-medvedeva/GADES-main to build |
| `GADES_CUDA_ARCH` | `sm_86` | compute capability of the GPU GADES is compiled for (`sm_86` is the RTX 3090) |
| `ARRAYFIRE_VERSION` | `v3.10.0` | tag of ArrayFire |
| `CRAN_MIRROR` | `https://cloud.r-project.org` | CRAN mirror for the R packages |

The published image is built with the defaults. 
For another GPU, for example an A100, build it with `docker build --build-arg GADES_CUDA_ARCH=sm_80 -t akhtyamovpavel/article-gades-2 .`

### Run the benchmark

The container starts in `scripts/Benchmarking`. Mount the datasets of Step 2 and a directory for new results, so that the published `results/` stay untouched:

```shell
mkdir -p my-results
docker run --rm --gpus all \
    -v $PWD/Datasets:/workspace/Article-GADES-2/Datasets:ro \
    -v $PWD/my-results:/workspace/Article-GADES-2/my-results \
    article-gades-2 \
    ./run_benchmark.sh real ../../Datasets/Real ../../my-results/RealDatasets
```

The other suites run the same way with `generated-dense ../../Datasets/GeneratedDense`, `generated-sparse ../../Datasets/GeneratedSparse` and `huge ../../Datasets/HugeDatasets`. Settings of Step 3 are passed with `-e`, for example `-e TIME_LIMIT=3600`. An interactive shell in the same environment:

```shell
docker run --rm -it --gpus all -v $PWD/Datasets:/workspace/Article-GADES-2/Datasets:ro article-gades-2 bash
```

The figures of Step 4 are drawn in the same image:

```shell
docker run --rm -v $PWD/figures:/workspace/Article-GADES-2/reproducibility/figures -w /workspace/Article-GADES-2 article-gades-2 python reproducibility/collect_real_results.py
```

## Step 2. Datasets

The matrices of the benchmark are published at https://huggingface.co/datasets/lab-medvedeva/GADES2.

```shell
pip install -U huggingface_hub
huggingface-cli download lab-medvedeva/GADES2 --repo-type dataset --local-dir Datasets
```

| Directory | Content |
|---|---|
| `Datasets/GeneratedDense` | 17 dense matrices `<cells>_cells_<features>_features.csv`, 10 to 10 000 cells and 10 to 100 000 features, at most 10^7 elements |
| `Datasets/GeneratedSparse` | 15 directories `<cells>_cells_<features>_features` with `0.5.mtx`, `0.75.mtx`, `0.9.mtx`, `0.95.mtx` and `0.99.mtx`, named by the share of zeros |
| `Datasets/Real` | 22 Matrix Market files `<dataset>.mtx` |
| `Datasets/HugeDatasets` | 10 AnnData files `<dataset>.h5ad` with a CSR matrix of objects by features |

Single-cell matrices of the real suite are stored as features by cells, recommender matrices as users by items; the benchmark script knows the orientation of each dataset.

GADES reads the huge datasets from h5ad directly. The baselines read Matrix Market, which is produced by

```shell
python scripts/Benchmarking/h5ad_to_mtx.py Datasets/HugeDatasets/<dataset>.h5ad Datasets/HugeDatasets/<dataset>.mtx
```

`scripts/MatricesGeneration` generates matrices of the same shapes with a fixed seed: `generate_dense.sh Datasets` writes `Datasets/Generated`, `generate_sparse.sh Datasets` writes `Datasets/GeneratedSparse`. Their values differ from the published matrices.

## Step 3. Benchmark

One script runs every case of a suite: one method, mode (dense or sparse) and metric on one dataset.

```shell
cd scripts/Benchmarking
./run_benchmark.sh generated-dense  ../../Datasets/GeneratedDense  ../../results/GeneratedDense
./run_benchmark.sh generated-sparse ../../Datasets/GeneratedSparse ../../results/GeneratedSparse
./run_benchmark.sh real             ../../Datasets/Real            ../../results/RealDatasets
./run_benchmark.sh huge             ../../Datasets/HugeDatasets    ../../results/HugeDatasets
```

Every case runs under a time limit: one day for the generated and the real suites, two days for the huge suite. A case that exceeds it leaves no result file. The output of a case goes to `<results>/logs/`.

Settings, all overridable through the environment:

| Variable | Default | Meaning |
|---|---|---|
| `TIME_LIMIT` | 86400, huge 172800 | time limit of one case, seconds |
| `TIMES` | see below | repetitions of one case |
| `BATCH_SIZE` | 5000 | batch size of GADES |
| `THREADS` | 24 | threads of the Python baselines |
| `MEM_LIMIT_GB` | 50 | memory limit of GADES, GB: below it the distance matrix is computed as one block, above it in batches, and a run that would exceed it stops |

Repetitions by default: generated dense 10 (Kendall 5, pandas and the Python loop 3), generated sparse 5, real and huge 2 for GADES on the GPU and for RAFT and 1 for the other methods. The first repetition of a GPU method includes the initialisation of CUDA.

### Methods

| Name in result files | What one repetition times |
|---|---|
| `GPU`, `CPU` | `mtrx_distance(data, batch_size, metric, type = "gpu"` or `"cpu", sparse, write = FALSE)` of GADES. Sparse Kendall, Spearman and Manhattan use the layout `per_cell_pair` |
| `amap` | `amap::Dist(x, method, nbproc = 24)`. The cosine case calls method `correlation`, the Pearson case calls method `pearson` of amap |
| `factoextra` | `factoextra::get_dist(x, method)`. The cosine case calls method `pearson`; the Pearson case is written to the cosine file, so there is no separate Pearson file |
| `pandas` | `DataFrame.corr(method)`: built-in Pearson, Spearman and Kendall, a Python function per pair for Euclidean (sum of squared differences), cosine (`(1 - numpy.corrcoef) / 2`) and Manhattan |
| `pythonic` | a loop over all pairs in a process pool (`tqdm.contrib.concurrent.process_map`) with the same per-pair functions and `scipy.stats` for Spearman and Kendall; Pearson is one `numpy.corrcoef` call |
| `raft` | `cuml.metrics.pairwise_distances` and `sparse_pairwise_distances`; Pearson is the cosine distance of centred rows; the result is copied back to the host |
| `armadillo` | matrix expressions of Armadillo, single precision |
| `arrayfirecuda`, `arrayfirecpu` | matrix expressions of ArrayFire; the input is uploaded and the result is copied back to the host inside the timed region |

Armadillo, ArrayFire and RAFT are run on Euclidean, cosine, Pearson and Manhattan. amap, factoextra, pandas and the Python loop take dense input only.

### Result files

A case writes `<results>/<dataset>/<mode>_<method>_<metric>.csv`: one row per repetition, time in microseconds. The generated sparse suite writes `<results>/<size>/<sparsity>/sparse_<method>_<metric>.csv`. The files in `results/` hold one to ten repetitions per case.

### Estimated cases

Some cases of the real and the huge suites were not run to completion. Their result files hold an estimate instead of a measurement and have a second column, `kind`, with the value `estimate`; a measured file has one column. In the real suite a case that exceeded the one-hour limit of the benchmark pass is estimated from the measured cases of the same method, mode and metric by a scaling law in the number of object pairs and the work per pair. In the huge suite GADES streams the input batch by batch, and a run longer than ten minutes is extrapolated from its finished batches. An estimated case has a file only when the estimate fits the time limit of the suite.

## Step 4. Figures

`reproducibility/README.md` lists the command behind every figure of the article. The collectors read the result files only: a case without a file counts as not finished within the time limit, and the time of a case is the median of its repetitions without the first one.

```shell
python3 reproducibility/collect_generated_dense.py
python3 reproducibility/collect_generated_sparse.py
python3 reproducibility/figure_generated_acceleration.py
python3 reproducibility/collect_real_results.py
python3 reproducibility/collect_huge_results.py
```

## Step 5. Accuracy

`scripts/Validation/README.md` describes the scripts that compare GADES with reference implementations.
