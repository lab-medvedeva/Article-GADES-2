#include <cstdlib>
// Armadillo benchmark for pairwise distance matrices
// Analogous to test_arrayfire.cpp — same CLI args, same output format
//
// Usage: ./test_armadillo <data_file> <backend> <times> <metric> <batch_size> <output> <sparse> [transpose]
//   backend: armadillo | armadillocpu
//   metric:  euclidean | cosine | pearson | manhattan | spearman
//   sparse:  TRUE | FALSE
//   transpose: TRUE | FALSE (default TRUE). Set FALSE for bio datasets where
//              MTX rows=features, cols=cells — already features×samples.

#include <armadillo>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <functional>

// ===================== Data container =====================
struct MatrixData {
    arma::fmat    dense;       // always available (features x samples)
    arma::sp_fmat sparse_csr;  // set only when sparse=true
    bool          has_sparse;
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
    arma::fmat result(n_features, n_samples);
    for (int s = 0; s < n_samples; s++) {
        for (int f = 0; f < n_features; f++) {
            result(f, s) = rows[s][f];
        }
    }

    std::cout << "Read dense CSV: " << n_features << " features x "
              << n_samples << " samples" << std::endl;
    return {result, arma::sp_fmat(), false};
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

    arma::fmat mat(nrows, ncols, arma::fill::zeros);
    for (int i = 0; i < nnz; i++) {
        int r, c; double v;
        file >> r >> c >> v;
        mat(r - 1, c - 1) = static_cast<float>(v);
    }

    if (do_transpose) {
        mat = mat.t();
        std::cout << "Read MTX (dense): " << ncols << " features x "
                  << nrows << " samples (nnz=" << nnz << ")" << std::endl;
    } else {
        std::cout << "Read MTX (dense): " << nrows << " features x "
                  << ncols << " samples (nnz=" << nnz << ")" << std::endl;
    }
    return {mat, arma::sp_fmat(), false};
}

// Read Matrix Market (.mtx) → sparse, plus a dense copy only when keep_dense is set
// (the sparse euclidean, cosine and Pearson paths never read the dense copy).
MatrixData read_mtx_sparse(const std::string& path, bool do_transpose = true, bool keep_dense = true) {
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

    arma::umat locations(2, nnz);
    arma::fvec values(nnz);
    for (int i = 0; i < nnz; i++) {
        int r, c; double v;
        file >> r >> c >> v;
        locations(0, i) = r - 1;
        locations(1, i) = c - 1;
        values(i) = static_cast<float>(v);
    }

    arma::sp_fmat sp_mat(locations, values, nrows, ncols);
    if (do_transpose) {
        arma::sp_fmat sp_result = sp_mat.t();
        arma::fmat result_dense;
        if (keep_dense) {
            result_dense = arma::fmat(sp_result);
        }
        std::cout << "Read MTX (sparse): " << ncols << " features x "
                  << nrows << " samples (nnz=" << nnz << ")" << std::endl;
        return {result_dense, sp_result, true};
    } else {
        arma::fmat result_dense;
        if (keep_dense) {
            result_dense = arma::fmat(sp_mat);
        }
        std::cout << "Read MTX (sparse): " << nrows << " features x "
                  << ncols << " samples (nnz=" << nnz << ")" << std::endl;
        return {result_dense, sp_mat, true};
    }
}

// ===================== Dense Distance Metrics =====================

// Euclidean: sqrt(sum((a-b)^2))
// ||a-b||^2 = ||a||^2 + ||b||^2 - 2*(a·b)
arma::fmat compute_euclidean(const MatrixData& md) {
    const arma::fmat& A = md.dense;
    int m = A.n_cols;
    arma::frowvec sq_norms = arma::sum(A % A, 0);  // (1, m)
    arma::fmat gram = A.t() * A;                     // (m, m)
    arma::fmat D = arma::repmat(sq_norms.t(), 1, m)
                 + arma::repmat(sq_norms, m, 1)
                 - 2.0f * gram;
    D = arma::clamp(D, 0.0f, D.max());
    return arma::sqrt(D);
}

// Cosine: 1 - dot(a,b) / (||a|| * ||b||)
arma::fmat compute_cosine(const MatrixData& md) {
    const arma::fmat& A = md.dense;
    arma::frowvec norms = arma::sqrt(arma::sum(A % A, 0));  // (1, m)
    arma::fmat gram = A.t() * A;                              // (m, m)
    arma::fmat norm_prod = norms.t() * norms;                 // (m, m)
    return 1.0f - gram / norm_prod;
}

// Pearson: 1 - corr(a,b) — center columns, then cosine
arma::fmat compute_pearson(const MatrixData& md) {
    const arma::fmat& A = md.dense;
    int n = A.n_rows;
    arma::frowvec means = arma::mean(A, 0);
    arma::fmat centered = A - arma::repmat(means, n, 1);

    arma::frowvec norms = arma::sqrt(arma::sum(centered % centered, 0));
    arma::fmat gram = centered.t() * centered;
    arma::fmat norm_prod = norms.t() * norms;
    return 1.0f - gram / norm_prod;
}

// Manhattan: sum(|a - b|) — loop over features
arma::fmat compute_manhattan(const MatrixData& md) {
    const arma::fmat& A = md.dense;
    int n = A.n_rows;
    int m = A.n_cols;

    arma::fmat D(m, m, arma::fill::zeros);
    for (int r = 0; r < n; r++) {
        arma::frowvec row = A.row(r);
        arma::fmat diff = arma::abs(arma::repmat(row.t(), 1, m) - arma::repmat(row, m, 1));
        D += diff;
    }
    return D;
}

// Spearman: rank data per column, then Pearson on ranks
arma::fmat compute_spearman(const MatrixData& md) {
    const arma::fmat& A = md.dense;
    int n = A.n_rows;
    int m = A.n_cols;

    arma::fmat ranks(n, m);
    for (int j = 0; j < m; j++) {
        arma::uvec sorted_idx = arma::sort_index(A.col(j));
        arma::fvec rank_col(n);
        for (int i = 0; i < n; i++) {
            rank_col(sorted_idx(i)) = static_cast<float>(i);
        }
        ranks.col(j) = rank_col;
    }

    MatrixData ranked_md = {ranks, arma::sp_fmat(), false};
    return compute_pearson(ranked_md);
}

// ===================== Sparse Distance Metrics =====================
// Use arma::sp_fmat for sparse matrix multiplication.

static arma::fmat sparse_gram(const MatrixData& md) {
    // A^T * A using sparse matrix
    return arma::fmat(md.sparse_csr.t() * md.sparse_csr);
}

// Column sums and squared norms straight from the sparse matrix.
static arma::frowvec sparse_col_sums(const MatrixData& md) {
    return arma::frowvec(arma::fmat(arma::sum(md.sparse_csr, 0)));
}

static arma::frowvec sparse_col_sq_norms(const MatrixData& md) {
    return arma::frowvec(arma::fmat(arma::sum(md.sparse_csr % md.sparse_csr, 0)));
}

arma::fmat compute_euclidean_sparse(const MatrixData& md) {
    int m = md.sparse_csr.n_cols;
    arma::frowvec sq_norms = sparse_col_sq_norms(md);
    arma::fmat gram = sparse_gram(md);
    arma::fmat D = arma::repmat(sq_norms.t(), 1, m)
                 + arma::repmat(sq_norms, m, 1)
                 - 2.0f * gram;
    D = arma::clamp(D, 0.0f, D.max());
    return arma::sqrt(D);
}

arma::fmat compute_cosine_sparse(const MatrixData& md) {
    arma::frowvec norms = arma::sqrt(sparse_col_sq_norms(md));
    arma::fmat gram = sparse_gram(md);
    arma::fmat norm_prod = norms.t() * norms;
    return 1.0f - gram / norm_prod;
}

// Pearson sparse: avoid explicit centering to preserve sparsity.
// corr(a,b) = (dot(a,b) - sum_a*sum_b/n) / (sqrt(sq_a - sum_a^2/n) * sqrt(sq_b - sum_b^2/n))
arma::fmat compute_pearson_sparse(const MatrixData& md) {
    int n = md.sparse_csr.n_rows;

    arma::frowvec col_sums = sparse_col_sums(md);
    arma::frowvec sq_norms = sparse_col_sq_norms(md);
    arma::fmat gram = sparse_gram(md);

    arma::fmat sum_prod = col_sums.t() * col_sums;
    arma::fmat dot_centered = gram - sum_prod / static_cast<float>(n);

    arma::frowvec norms = arma::sqrt(arma::clamp(
        sq_norms - col_sums % col_sums / static_cast<float>(n),
        0.0f, std::numeric_limits<float>::max()));
    arma::fmat norm_prod = norms.t() * norms;

    return 1.0f - dot_centered / norm_prod;
}

arma::fmat compute_manhattan_sparse(const MatrixData& md) {
    return compute_manhattan(md);
}

arma::fmat compute_spearman_sparse(const MatrixData& md) {
    return compute_spearman(md);
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
              << " <data_file> <backend:armadillo> <times> <metric>"
              << " <batch_size> <output> [sparse:TRUE|FALSE]"
              << std::endl;
    std::cerr << "Metrics: euclidean, cosine, pearson, manhattan, spearman" << std::endl;
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

    std::cout << "Data:       " << data_path   << std::endl;
    std::cout << "Backend:    " << backend_str  << std::endl;
    std::cout << "Times:      " << times        << std::endl;
    std::cout << "Metric:     " << metric       << std::endl;
    std::cout << "Batch size: " << batch_size   << std::endl;
    std::cout << "Output:     " << output       << std::endl;
    std::cout << "Sparse:     " << sparse       << std::endl;
    std::cout << "Transpose:  " << do_transpose << std::endl;

    std::string method_name = "armadillo";

    // ---------- read data ----------
    std::cout << "Reading data..." << std::endl;
    MatrixData md;
    if (sparse) {
        // Only manhattan and spearman fall back to the dense functions in sparse mode.
        bool keep_dense = !(metric == "euclidean" || metric == "cosine" || metric == "pearson");
        md = read_mtx_sparse(data_path, do_transpose, keep_dense);
    } else {
        if (data_path.size() >= 4 &&
            data_path.substr(data_path.size() - 4) == ".mtx") {
            md = read_mtx_dense(data_path, do_transpose);
        } else {
            md = read_dense_csv(data_path);
        }
    }
    const arma::uword in_rows = md.has_sparse ? md.sparse_csr.n_rows : md.dense.n_rows;
    const arma::uword in_cols = md.has_sparse ? md.sparse_csr.n_cols : md.dense.n_cols;
    std::cout << "Input shape:  " << in_rows << " x " << in_cols
              << "  (rows=features, cols=samples)" << std::endl;
    std::cout << "Output shape: " << in_cols << " x " << in_cols
              << "  (samples x samples)" << std::endl;
    std::cout << "Dense copy:   " << (md.dense.n_elem > 0 ? "yes" : "no") << std::endl;
    if (md.has_sparse) {
        std::cout << "Sparse:       nnz=" << md.sparse_csr.n_nonzero << std::endl;
    }

    // ---------- select metric ----------
    typedef std::function<arma::fmat(const MatrixData&)> DistFunc;
    DistFunc compute_distance;

    if (sparse && md.has_sparse) {
        if      (metric == "euclidean")  compute_distance = compute_euclidean_sparse;
        else if (metric == "cosine")     compute_distance = compute_cosine_sparse;
        else if (metric == "pearson")    compute_distance = compute_pearson_sparse;
        else if (metric == "manhattan")  compute_distance = compute_manhattan_sparse;
        else if (metric == "spearman")   compute_distance = compute_spearman_sparse;
        else {
            std::cerr << "Unsupported metric: " << metric << std::endl;
            return 1;
        }
        std::cout << "Mode:         SPARSE" << std::endl;
    } else {
        if      (metric == "euclidean")  compute_distance = compute_euclidean;
        else if (metric == "cosine")     compute_distance = compute_cosine;
        else if (metric == "pearson")    compute_distance = compute_pearson;
        else if (metric == "manhattan")  compute_distance = compute_manhattan;
        else if (metric == "spearman")   compute_distance = compute_spearman;
        else {
            std::cerr << "Unsupported metric: " << metric << std::endl;
            return 1;
        }
        std::cout << "Mode:         DENSE" << std::endl;
    }

    // ---------- warmup ----------
    {
        arma::fmat warmup = compute_distance(md);
        if (std::getenv("ARMA_CHECKSUM") != nullptr) {
            arma::uvec finite = arma::find_finite(warmup);
            double total = arma::accu(arma::conv_to<arma::vec>::from(warmup.elem(finite)));
            std::cout << "Checksum:     finite=" << finite.n_elem << " sum=" << total
                      << " d01=" << warmup(0, 1) << " d12=" << warmup(1, 2) << std::endl;
        }
        (void)warmup;
    }

    // ---------- benchmark ----------
    std::vector<double> measurements(times);
    std::string out_file = output + "_" + method_name + "_" + metric + ".csv";

    for (int i = 0; i < times; i++) {
        auto t0 = std::chrono::high_resolution_clock::now();

        arma::fmat dist = compute_distance(md);
        (void)dist.n_elem;  // ensure evaluation

        auto t1 = std::chrono::high_resolution_clock::now();
        double elapsed_us =
            std::chrono::duration<double, std::micro>(t1 - t0).count();

        measurements[i] = elapsed_us;
        std::cout << "  [" << (i + 1) << "/" << times << "] "
                  << elapsed_us << " us  (" << elapsed_us / 1e6 << " s)"
                  << "  result: " << dist.n_rows << "x" << dist.n_cols
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
