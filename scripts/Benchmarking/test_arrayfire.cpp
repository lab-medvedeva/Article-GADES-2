// ArrayFire benchmark for pairwise distance matrices
// Analogous to test.R — same CLI args, same output format
//
// Usage: ./test_arrayfire <data_file> <backend> <times> <metric> <batch_size> <output> <sparse> [transpose]
//   backend: arrayfirecuda | arrayfirecpu
//   metric:  euclidean | cosine | pearson | manhattan | spearman
//            euclidean_nn | manhattan_nn | ssd_nn  (via af::nearestNeighbour)
//   sparse:  TRUE | FALSE
//   transpose: TRUE | FALSE (default TRUE). Set FALSE for bio datasets where
//              MTX rows=features, cols=cells — already features×samples.
//
// When sparse=TRUE, reads MTX as af::sparse (CSR) and uses sparse matmul
// (cuSPARSE on CUDA, parallel on CPU) for Gram matrix computation.

#include <arrayfire.h>
#include <af/sparse.h>
#include <af/vision.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <functional>

using namespace af;

// ===================== Data container =====================
// Holds dense representation + optional sparse CSR for matmul
struct MatrixData {
    array dense;          // always available (features x samples)
    array sparse_csr;     // af::sparse CSR, set only when sparse=true
    bool  has_sparse;
};

// ===================== Data I/O =====================

// Read dense CSV: first row = header, first col = row names, comma-separated
// Returns (features x samples) — same as R's t(read.table(...))
MatrixData read_dense_csv(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << path << std::endl;
        exit(1);
    }

    std::string line;
    std::getline(file, line);  // skip header

    std::vector<std::vector<float>> rows;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        std::string token;
        std::getline(iss, token, ',');  // skip row name

        std::vector<float> row;
        while (std::getline(iss, token, ',')) {
            try {
                row.push_back(std::stof(token));
            } catch (...) {
                row.push_back(0.0f);
            }
        }
        rows.push_back(row);
    }

    int n_samples  = static_cast<int>(rows.size());
    int n_features = static_cast<int>(rows[0].size());

    // CSV is (samples x features), we want (features x samples)
    std::vector<float> flat(n_features * n_samples);
    for (int s = 0; s < n_samples; s++) {
        for (int f = 0; f < n_features; f++) {
            flat[f + s * n_features] = rows[s][f];
        }
    }

    array result(n_features, n_samples, flat.data());
    std::cout << "Read dense CSV: " << n_features << " features x "
              << n_samples << " samples" << std::endl;
    return {result, array(), false};
}

// Read Matrix Market (.mtx) → dense only (sparse=FALSE path)
MatrixData read_mtx_dense(const std::string& path, bool do_transpose = true) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << path << std::endl;
        exit(1);
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line[0] != '%') break;
    }

    std::istringstream dims(line);
    int nrows, ncols, nnz;
    dims >> nrows >> ncols >> nnz;

    std::vector<int>   row_idx(nnz), col_idx(nnz);
    std::vector<float> values(nnz);

    for (int i = 0; i < nnz; i++) {
        int r, c; double v;
        file >> r >> c >> v;
        row_idx[i] = r - 1;
        col_idx[i] = c - 1;
        values[i]  = static_cast<float>(v);
    }

    std::vector<float> dense_buf(static_cast<size_t>(nrows) * ncols, 0.0f);
    for (int i = 0; i < nnz; i++) {
        dense_buf[row_idx[i] + static_cast<size_t>(col_idx[i]) * nrows] = values[i];
    }

    array mat(nrows, ncols, dense_buf.data());
    if (do_transpose) {
        mat = transpose(mat);
        std::cout << "Read MTX (dense): " << ncols << " features x "
                  << nrows << " samples (nnz=" << nnz << ")" << std::endl;
    } else {
        std::cout << "Read MTX (dense): " << nrows << " features x "
                  << ncols << " samples (nnz=" << nnz << ")" << std::endl;
    }
    return {mat, array(), false};
}

// Read Matrix Market (.mtx) → af::sparse CSR + dense copy
// Sparse matmul requires: sparse LHS (CSR) × dense RHS
// So we keep both representations.
MatrixData read_mtx_sparse(const std::string& path, bool do_transpose = true) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << path << std::endl;
        exit(1);
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line[0] != '%') break;
    }

    std::istringstream dims_ss(line);
    int nrows, ncols, nnz;
    dims_ss >> nrows >> ncols >> nnz;

    std::vector<int>   row_idx(nnz), col_idx(nnz);
    std::vector<float> values(nnz);

    for (int i = 0; i < nnz; i++) {
        int r, c; double v;
        file >> r >> c >> v;
        row_idx[i] = r - 1;
        col_idx[i] = c - 1;
        values[i]  = static_cast<float>(v);
    }

    // Build dense (nrows x ncols)
    std::vector<float> dense_buf(static_cast<size_t>(nrows) * ncols, 0.0f);
    for (int i = 0; i < nnz; i++) {
        dense_buf[row_idx[i] + static_cast<size_t>(col_idx[i]) * nrows] = values[i];
    }
    array mat_dense(nrows, ncols, dense_buf.data());

    if (do_transpose) {
        // Transpose both: t(readMM(...))
        array result_dense = transpose(mat_dense);
        array result_sparse = af::sparse(result_dense, AF_STORAGE_CSR);

        dim_t out_nnz = af::sparseGetNNZ(result_sparse);
        std::cout << "Read MTX (sparse CSR): " << ncols << " features x "
                  << nrows << " samples (nnz=" << nnz
                  << ", after transpose nnz=" << out_nnz << ")" << std::endl;

        return {result_dense, result_sparse, true};
    } else {
        // No transpose: MTX already has features×samples
        array result_sparse = af::sparse(mat_dense, AF_STORAGE_CSR);

        dim_t out_nnz = af::sparseGetNNZ(result_sparse);
        std::cout << "Read MTX (sparse CSR): " << nrows << " features x "
                  << ncols << " samples (nnz=" << nnz
                  << ", sparse nnz=" << out_nnz << ")" << std::endl;

        return {mat_dense, result_sparse, true};
    }
}

// ===================== Dense Distance Metrics =====================

// Euclidean: sqrt(sum((a-b)^2))
// ||a-b||^2 = ||a||^2 + ||b||^2 - 2*(a·b)
array compute_euclidean(const MatrixData& md) {
    const array& A = md.dense;
    int m = A.dims(1);
    array sq_norms = sum(A * A, 0);          // (1, m)
    array gram     = matmulTN(A, A);         // (m, m)
    array D = tile(transpose(sq_norms), 1, m)
            + tile(sq_norms, m, 1)
            - 2.0f * gram;
    return sqrt(af::max(D, 0.0f));
}

// Cosine: 1 - dot(a,b) / (||a|| * ||b||)
array compute_cosine(const MatrixData& md) {
    const array& A = md.dense;
    array norms     = sqrt(sum(A * A, 0));           // (1, m)
    array gram      = matmulTN(A, A);                // (m, m)
    array norm_prod = matmul(transpose(norms), norms); // (m, m)
    return 1.0f - gram / norm_prod;
}

// Pearson: 1 - corr(a,b)  — center columns, then cosine
array compute_pearson(const MatrixData& md) {
    const array& A = md.dense;
    int n = A.dims(0);
    array means    = mean(A, 0);
    array centered = A - tile(means, n, 1);

    array norms     = sqrt(sum(centered * centered, 0));
    array gram      = matmulTN(centered, centered);
    array norm_prod = matmul(transpose(norms), norms);
    return 1.0f - gram / norm_prod;
}

// Manhattan: sum(|a - b|)  — loop over features
array compute_manhattan(const MatrixData& md) {
    const array& A = md.dense;
    int n = A.dims(0);
    int m = A.dims(1);

    array D = constant(0.0f, m, m);
    for (int r = 0; r < n; r++) {
        array row = A(r, span);
        array diff = abs(tile(transpose(row), 1, m) - tile(row, m, 1));
        D += diff;
    }
    return D;
}

// Spearman: rank data per column, then Pearson on ranks
array compute_spearman(const MatrixData& md) {
    const array& A = md.dense;
    array vals, idx, vals2, ranks;
    sort(vals, idx, A, 0);
    sort(vals2, ranks, idx, 0);
    MatrixData ranked_md = {ranks.as(f32), array(), false};
    return compute_pearson(ranked_md);
}

// ===================== Sparse Distance Metrics =====================
// Use af::sparse CSR for matmul (cuSPARSE / parallel CPU).
// Sparse matmul: matmul(sparse_LHS, dense_RHS, AF_MAT_TRANS) = A^T * A via cuSPARSE

static array sparse_gram(const MatrixData& md) {
    // matmul(sparse_A, dense_A, AF_MAT_TRANS, AF_MAT_NONE) = A^T * A
    return af::matmul(md.sparse_csr, md.dense, AF_MAT_TRANS, AF_MAT_NONE);
}

array compute_euclidean_sparse(const MatrixData& md) {
    const array& A = md.dense;
    int m = A.dims(1);
    array sq_norms = sum(A * A, 0);          // (1, m)
    array gram     = sparse_gram(md);        // (m, m) via cuSPARSE
    array D = tile(transpose(sq_norms), 1, m)
            + tile(sq_norms, m, 1)
            - 2.0f * gram;
    return sqrt(af::max(D, 0.0f));
}

array compute_cosine_sparse(const MatrixData& md) {
    const array& A = md.dense;
    array norms     = sqrt(sum(A * A, 0));
    array gram      = sparse_gram(md);
    array norm_prod = matmul(transpose(norms), norms);
    return 1.0f - gram / norm_prod;
}

// Pearson sparse: avoid explicit centering to preserve sparsity.
// Same formula as FinalizePearsonSparse in main.cu:
//   corr(a,b) = (dot(a,b) - sum_a*sum_b/n) / (sqrt(sq_a - sum_a^2/n) * sqrt(sq_b - sum_b^2/n))
// dot(a,b) = Gram matrix → sparse matmul via cuSPARSE
// sum, sq = computed from dense (cheap 1D reductions)
array compute_pearson_sparse(const MatrixData& md) {
    const array& A = md.dense;
    int n = A.dims(0);  // features
    int m = A.dims(1);  // samples

    array col_sums = sum(A, 0);              // (1, m)
    array sq_norms = sum(A * A, 0);          // (1, m)
    array gram     = sparse_gram(md);        // (m, m) via cuSPARSE

    // dot_centered[i,j] = gram[i,j] - col_sums[i]*col_sums[j] / n
    array sum_prod = matmul(transpose(col_sums), col_sums);  // (m, m)
    array dot_centered = gram - sum_prod / static_cast<float>(n);

    // norm[i] = sqrt(sq_norms[i] - col_sums[i]^2 / n)
    array norms = sqrt(af::max(sq_norms - col_sums * col_sums / static_cast<float>(n), 0.0f));
    array norm_prod = matmul(transpose(norms), norms);  // (m, m)

    return 1.0f - dot_centered / norm_prod;
}

// Manhattan sparse: no matmul shortcut for L1, same parallel loop as dense
array compute_manhattan_sparse(const MatrixData& md) {
    return compute_manhattan(md);
}

// Spearman sparse: rank each column (destroys sparsity), then Pearson on ranks.
// Ranking + dense Pearson — ranking itself is parallel (sort on GPU/CPU).
array compute_spearman_sparse(const MatrixData& md) {
    return compute_spearman(md);
}

// ===================== nearestNeighbour-based metrics =====================
// Uses ArrayFire's internal all_distances kernel (AF_SSD / AF_SAD).
// n_dist is capped at min(n_samples, 256) by ArrayFire.

static array nn_distances(const MatrixData& md, af_match_type dist_type) {
    const array& A = md.dense;
    dim_t m = A.dims(1);
    unsigned n_dist = static_cast<unsigned>(std::min(m, dim_t(256)));
    array idx, dist;
    af::nearestNeighbour(idx, dist, A, A, 0, n_dist, dist_type);
    return dist;
}

array compute_euclidean_nn(const MatrixData& md) {
    return sqrt(af::max(nn_distances(md, AF_SSD), 0.0f));
}

array compute_manhattan_nn(const MatrixData& md) {
    return nn_distances(md, AF_SAD);
}

array compute_ssd_nn(const MatrixData& md) {
    return nn_distances(md, AF_SSD);
}

// ===================== CSV Output =====================

void write_measurements(const std::string& path, const std::vector<double>& m) {
    std::ofstream ofs(path);
    ofs << "\"x\"" << std::endl;
    for (size_t i = 0; i < m.size(); i++) {
        ofs << "\"" << (i + 1) << "\"," << m[i] << std::endl;
    }
}

// ===================== Main =====================

void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog
              << " <data_file> <backend:arrayfirecuda|arrayfirecpu> <times> <metric>"
              << " <batch_size> <output> [sparse:TRUE|FALSE]"
              << std::endl;
    std::cerr << "Metrics: euclidean, cosine, pearson, manhattan, spearman" << std::endl;
    std::cerr << "         euclidean_nn, manhattan_nn, ssd_nn (via nearestNeighbour)"
              << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 7) {
        print_usage(argv[0]);
        return 1;
    }

    std::string data_path   = argv[1];
    std::string backend_str = argv[2];
    int         times       = std::stoi(argv[3]);
    std::string metric      = argv[4];
    int         batch_size  = std::stoi(argv[5]);
    std::string output      = argv[6];

    bool sparse = false;
    if (argc >= 8) {
        std::string s = argv[7];
        sparse = (s == "TRUE" || s == "true" || s == "1");
    }

    bool do_transpose = true;
    if (argc >= 9) {
        std::string s = argv[8];
        do_transpose = !(s == "FALSE" || s == "false" || s == "0");
    }

    // arg9 = roundtrip: when TRUE, the timed region re-uploads the host input
    // (H2D) AND copies the result back to host (D2H) every iteration — the same
    // host->host scope GADES/armadillo/raft measure. Default FALSE = compute-only
    // (data resident, result left on device): the original kernel-throughput mode.
    bool roundtrip = false;
    if (argc >= 10) {
        std::string s = argv[9];
        roundtrip = (s == "TRUE" || s == "true" || s == "1");
    }

    std::cout << "Data:       " << data_path   << std::endl;
    std::cout << "Backend:    " << backend_str  << std::endl;
    std::cout << "Times:      " << times        << std::endl;
    std::cout << "Metric:     " << metric       << std::endl;
    std::cout << "Batch size: " << batch_size   << std::endl;
    std::cout << "Output:     " << output       << std::endl;
    std::cout << "Sparse:     " << sparse       << std::endl;
    std::cout << "Transpose:  " << do_transpose << std::endl;

    // ---------- backend ----------
    std::string method_name;
    try {
        if (backend_str == "arrayfirecuda" || backend_str == "cuda" ||
            backend_str == "CUDA" || backend_str == "GPU") {
            af::setBackend(AF_BACKEND_CUDA);
            method_name = "arrayfirecuda";
        } else if (backend_str == "arrayfirecpu" || backend_str == "cpu" ||
                   backend_str == "CPU") {
            af::setBackend(AF_BACKEND_CPU);
            method_name = "arrayfirecpu";
        } else {
            std::cerr << "Unknown backend: " << backend_str << std::endl;
            return 1;
        }
    } catch (af::exception& e) {
        std::cerr << "Failed to set backend: " << e.what() << std::endl;
        return 1;
    }

    af::info();
    std::cout << std::endl;

    // ---------- read data ----------
    std::cout << "Reading data..." << std::endl;
    MatrixData md;
    if (sparse) {
        md = read_mtx_sparse(data_path, do_transpose);
    } else {
        if (data_path.size() >= 4 &&
            data_path.substr(data_path.size() - 4) == ".mtx") {
            md = read_mtx_dense(data_path, do_transpose);
        } else {
            md = read_dense_csv(data_path);
        }
    }
    std::cout << "Input shape:  " << md.dense.dims(0) << " x " << md.dense.dims(1)
              << "  (rows=features, cols=samples)" << std::endl;
    std::cout << "Output shape: " << md.dense.dims(1) << " x " << md.dense.dims(1)
              << "  (samples x samples)" << std::endl;
    if (md.has_sparse) {
        std::cout << "Sparse CSR:   nnz=" << af::sparseGetNNZ(md.sparse_csr)
                  << "  storage=CSR" << std::endl;
    }

    // ---------- select metric ----------
    typedef std::function<array(const MatrixData&)> DistFunc;
    DistFunc compute_distance;

    if (sparse && md.has_sparse) {
        // Sparse-aware versions (use cuSPARSE for Gram where beneficial)
        if      (metric == "euclidean")  compute_distance = compute_euclidean_sparse;
        else if (metric == "cosine")     compute_distance = compute_cosine_sparse;
        else if (metric == "pearson")    compute_distance = compute_pearson_sparse;
        else if (metric == "manhattan")  compute_distance = compute_manhattan_sparse;
        else if (metric == "spearman")   compute_distance = compute_spearman_sparse;
        else if (metric == "euclidean_nn")   compute_distance = compute_euclidean_nn;
        else if (metric == "manhattan_nn")   compute_distance = compute_manhattan_nn;
        else if (metric == "ssd_nn")         compute_distance = compute_ssd_nn;
        else {
            std::cerr << "Unsupported metric: " << metric << std::endl;
            return 1;
        }
        std::cout << "Mode:         SPARSE (cuSPARSE matmul)" << std::endl;
    } else {
        // Dense versions
        if      (metric == "euclidean")      compute_distance = compute_euclidean;
        else if (metric == "cosine")         compute_distance = compute_cosine;
        else if (metric == "pearson")        compute_distance = compute_pearson;
        else if (metric == "manhattan")      compute_distance = compute_manhattan;
        else if (metric == "spearman")       compute_distance = compute_spearman;
        else if (metric == "euclidean_nn")   compute_distance = compute_euclidean_nn;
        else if (metric == "manhattan_nn")   compute_distance = compute_manhattan_nn;
        else if (metric == "ssd_nn")         compute_distance = compute_ssd_nn;
        else {
            std::cerr << "Unsupported metric: " << metric << std::endl;
            return 1;
        }
        std::cout << "Mode:         DENSE" << std::endl;
    }

    // ---------- warmup ----------
    {
        array warmup = compute_distance(md);
        af::eval(warmup);
        af::sync();
    }

    // ---------- benchmark ----------
    std::vector<double> measurements(times);
    std::string out_file = output + "_" + method_name + "_" + metric + ".csv";
    std::cout << "Mode timing: " << (roundtrip ? "ROUND-TRIP (H2D + compute + D2H)"
                                               : "compute-only (resident)") << std::endl;

    // For round-trip: keep a host copy of the dense input to re-upload each iter.
    std::vector<float> h_dense;
    dim_t hd0 = md.dense.dims(0), hd1 = md.dense.dims(1);
    if (roundtrip) { h_dense.resize(md.dense.elements()); md.dense.host(h_dense.data()); }

    for (int i = 0; i < times; i++) {
        af::sync();
        auto t0 = std::chrono::high_resolution_clock::now();

        array dist;
        if (roundtrip) {
            array A = af::array(hd0, hd1, h_dense.data());     // H2D upload
            MatrixData md_i = md;
            md_i.dense = A;
            if (md.has_sparse) { md_i.sparse_csr = af::sparse(A, AF_STORAGE_CSR); }
            dist = compute_distance(md_i);
            af::eval(dist);
            std::vector<float> host_out(dist.elements());
            dist.host(host_out.data());                        // D2H result -> host
        } else {
            dist = compute_distance(md);
            af::eval(dist);
        }
        af::sync();

        auto t1 = std::chrono::high_resolution_clock::now();
        double elapsed_us =
            std::chrono::duration<double, std::micro>(t1 - t0).count();

        measurements[i] = elapsed_us;
        std::cout << "  [" << (i + 1) << "/" << times << "] "
                  << elapsed_us << " us  (" << elapsed_us / 1e6 << " s)"
                  << std::endl;

        write_measurements(out_file, measurements);
    }

    // ---------- stats ----------
    if (times > 1) {
        double sum = 0, sum_sq = 0;
        int cnt = times - 1;
        for (int i = 1; i < times; i++) {
            sum    += measurements[i];
            sum_sq += measurements[i] * measurements[i];
        }
        double avg = sum / cnt;
        double sd  = std::sqrt(std::max(sum_sq / cnt - avg * avg, 0.0));
        std::cout << "\nStats (excl. warmup iteration 0):" << std::endl;
        std::cout << "  Mean: " << avg << " us  (" << avg / 1e6 << " s)" << std::endl;
        std::cout << "  Std:  " << sd  << " us  (" << sd  / 1e6 << " s)" << std::endl;
    }

    std::cout << "Results saved to " << out_file << std::endl;
    return 0;
}
