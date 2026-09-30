// Armadillo Spearman distance dumper for spearman_ties_sweep.R. Computes the
// Spearman distance with the same expressions as compute_spearman and
// compute_pearson in scripts/Benchmarking/test_armadillo.cpp (ranks 0..n-1
// from sort_index, without averaging over ties) and writes the full matrix to
// a CSV file. Needs Armadillo; no GPU.
//
// Build:  g++ -O3 -std=c++17 armadillo_spearman_dump.cpp -o armadillo_spearman_dump -larmadillo
//
// Usage: ./armadillo_spearman_dump <input.csv> <output.csv>
//   input.csv:  a header row, then one row per object: name,v1,v2,...
//               (objects x features)
//   output.csv: objects x objects distance matrix without a header

#include <armadillo>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <string>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: " << argv[0] << " in.csv out.csv\n";
        return 1;
    }
    std::ifstream file(argv[1]);
    if (!file.is_open()) {
        std::cerr << "cannot open\n";
        return 1;
    }
    std::string line;
    // skip the header
    std::getline(file, line);
    std::vector<std::vector<float>> rows;
    while (std::getline(file, line)) {
        if (line.empty()) {
            continue;
        }
        std::istringstream iss(line);
        std::string tok;
        // skip the row name
        std::getline(iss, tok, ',');
        std::vector<float> row;
        while (std::getline(iss, tok, ',')) {
            try {
                row.push_back(std::stof(tok));
            } catch (...) {
                row.push_back(0.0f);
            }
        }
        rows.push_back(row);
    }
    int n_samples = (int)rows.size();
    int n_features = (int)rows[0].size();
    // features x objects: columns are the objects
    arma::fmat A(n_features, n_samples);
    for (int s = 0; s < n_samples; s++) {
        for (int f = 0; f < n_features; f++) {
            A(f, s) = rows[s][f];
        }
    }

    // ranks, the expressions of compute_spearman in test_armadillo.cpp
    int n = A.n_rows;
    int m = A.n_cols;
    arma::fmat ranks(n, m);
    for (int j = 0; j < m; j++) {
        arma::uvec sorted_idx = arma::sort_index(A.col(j));
        arma::fvec rank_col(n);
        for (int i = 0; i < n; i++) {
            rank_col(sorted_idx(i)) = (float)i;
        }
        ranks.col(j) = rank_col;
    }
    // Pearson distance of the ranks, the expressions of compute_pearson
    arma::frowvec means = arma::mean(ranks, 0);
    arma::fmat centered = ranks - arma::repmat(means, n, 1);
    arma::frowvec norms = arma::sqrt(arma::sum(centered % centered, 0));
    arma::fmat gram = centered.t() * centered;
    arma::fmat norm_prod = norms.t() * norms;
    arma::fmat D = 1.0f - gram / norm_prod;

    std::ofstream ofs(argv[2]);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            ofs << D(i, j);
            if (j + 1 < m) {
                ofs << ",";
            }
        }
        ofs << "\n";
    }
    std::cerr << "wrote " << m << "x" << m << " armadillo spearman to " << argv[2] << "\n";
    return 0;
}
