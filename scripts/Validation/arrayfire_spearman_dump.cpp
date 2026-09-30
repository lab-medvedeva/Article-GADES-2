// ArrayFire Spearman distance dumper for spearman_ties.R and
// spearman_ties_sweep.R. Computes the Spearman distance with the same
// expressions as compute_spearman and compute_pearson in
// scripts/Benchmarking/test_arrayfire.cpp (ranks from a sort of the sort
// permutation, without averaging over ties) and writes the full matrix to a
// CSV file. Needs ArrayFire with the CUDA backend.
//
// Build (ArrayFire installed system-wide):
//   g++ -O3 -std=c++17 arrayfire_spearman_dump.cpp -o arrayfire_spearman_dump -laf
// Build (ArrayFire build tree in $AF):
//   g++ -O3 -std=c++17 -I$AF/include -I$AF/build/include \
//       arrayfire_spearman_dump.cpp -o arrayfire_spearman_dump \
//       -L$AF/build/src/api/unified -laf
//
// Usage: ./arrayfire_spearman_dump <input.csv> <output.csv>
//   input.csv:  a header row, then one row per object: name,v1,v2,...
//               (objects x features, the reader of test_arrayfire.cpp)
//   output.csv: objects x objects distance matrix without a header

#include <arrayfire.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <string>

using namespace af;

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: " << argv[0] << " in.csv out.csv\n";
        return 1;
    }
    std::string in = argv[1];
    std::string out = argv[2];
    setBackend(AF_BACKEND_CUDA);

    std::ifstream file(in);
    if (!file.is_open()) {
        std::cerr << "cannot open " << in << "\n";
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
    std::vector<float> flat(n_features * n_samples);
    for (int s = 0; s < n_samples; s++) {
        for (int f = 0; f < n_features; f++) {
            flat[f + s * n_features] = rows[s][f];
        }
    }
    // features x objects: columns are the objects
    array A(n_features, n_samples, flat.data());

    // ranks (compute_spearman) and Pearson distance of the ranks (compute_pearson)
    array vals;
    array idx;
    array vals2;
    array ranks;
    sort(vals, idx, A, 0);
    sort(vals2, ranks, idx, 0);
    array R = ranks.as(f32);
    int n = R.dims(0);
    array means = mean(R, 0);
    array centered = R - tile(means, n, 1);
    array norms = sqrt(sum(centered * centered, 0));
    array gram = matmulTN(centered, centered);
    array norm_prod = matmul(transpose(norms), norms);
    array D = 1.0f - gram / norm_prod;
    eval(D);
    sync();

    int m = D.dims(0);
    std::vector<float> h(m * m);
    // column-major copy to the host
    D.host(h.data());
    std::ofstream ofs(out);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            ofs << h[i + j * m];
            if (j + 1 < m) {
                ofs << ",";
            }
        }
        ofs << "\n";
    }
    std::cerr << "wrote " << m << "x" << m << " spearman matrix to " << out << "\n";
    return 0;
}
