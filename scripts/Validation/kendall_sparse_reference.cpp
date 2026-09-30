// Sparse Kendall: cross-validation of the O(k log k) discordant count used by
// the per-cell-pair sparse Kendall kernel of GADES (k = number of features
// that are nonzero in at least one of the two objects).
//
// For every pair of objects the Fenwick-tree count is compared with:
//   (1) dense naive   - O(N^2) over all features, zeros included: the
//                       definition of the discordant count;
//   (2) sparse brute  - O(k^2) double two-pointer merge over the nonzero
//                       features plus the closed form n_signflip * n_inactive
//                       for the features that are zero in both objects.
//
// The data are integer-valued, so the criterion (a_i - a_j) * (b_i - b_j) < 0
// is exact in single precision. Standalone: needs neither GADES nor a GPU.
// Exit status is 0 when there are no mismatches.
//
// Build:  g++ -O2 -std=c++17 kendall_sparse_reference.cpp -o kendall_sparse_reference
// Run:    ./kendall_sparse_reference

#include <cstdio>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>

using namespace std;

// (1) Dense naive count: O(N^2) over all features, zeros included.
static long long disc_dense(const vector<float>& a, const vector<float>& b) {
    int N = (int)a.size();
    long long d = 0;
    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            if ((a[i] - a[j]) * (b[i] - b[j]) < 0.0f) {
                ++d;
            }
        }
    }
    return d;
}

// (2) Sparse brute force: O(k^2) double merge over the active features plus
//     the closed form for the block that is zero in both objects.
static long long disc_sparse_brute(
    const int* a_i, const float* a_x, int ia, int ea,
    const int* b_i, const float* b_x, int ib, int eb,
    int n_genes
) {
    long long discordant = 0;
    int n_active = 0;
    int n_signflip = 0;
    int oia = ia;
    int oib = ib;
    while (oia < ea || oib < eb) {
        float a_v;
        float b_v;
        int nia = oia;
        int nib = oib;
        if (oia < ea && (oib >= eb || a_i[oia] < b_i[oib])) {
            a_v = a_x[oia];
            b_v = 0.0f;
            nia = oia + 1;
        } else if (oib < eb && (oia >= ea || b_i[oib] < a_i[oia])) {
            a_v = 0.0f;
            b_v = b_x[oib];
            nib = oib + 1;
        } else {
            a_v = a_x[oia];
            b_v = b_x[oib];
            nia = oia + 1;
            nib = oib + 1;
        }
        ++n_active;
        if (a_v * b_v < 0.0f) {
            ++n_signflip;
        }
        int ja = nia;
        int jb = nib;
        while (ja < ea || jb < eb) {
            float aj;
            float bj;
            if (ja < ea && (jb >= eb || a_i[ja] < b_i[jb])) {
                aj = a_x[ja];
                bj = 0.0f;
                ++ja;
            } else if (jb < eb && (ja >= ea || b_i[jb] < a_i[ja])) {
                aj = 0.0f;
                bj = b_x[jb];
                ++jb;
            } else {
                aj = a_x[ja];
                bj = b_x[jb];
                ++ja;
                ++jb;
            }
            if ((a_v - aj) * (b_v - bj) < 0.0f) {
                ++discordant;
            }
        }
        oia = nia;
        oib = nib;
    }
    long long n_inactive = (long long)n_genes - n_active;
    discordant += (long long)n_signflip * n_inactive;
    return discordant;
}

// (3) Fenwick-tree count, O(k log k). Same algorithm as
//     kendall_per_cell_pair_merge, the per-cell-pair sparse Kendall kernel of
//     GADES.
static long long disc_sparse_fenwick(
    const int* a_i, const float* a_x, int ia, int ea,
    const int* b_i, const float* b_x, int ib, int eb,
    int n_genes
) {
    vector<float> av;
    vector<float> bv;
    av.reserve((ea - ia) + (eb - ib));
    bv.reserve((ea - ia) + (eb - ib));
    int n_signflip = 0;
    int oia = ia;
    int oib = ib;
    while (oia < ea || oib < eb) {
        float a_v;
        float b_v;
        if (oia < ea && (oib >= eb || a_i[oia] < b_i[oib])) {
            a_v = a_x[oia];
            b_v = 0.0f;
            ++oia;
        } else if (oib < eb && (oia >= ea || b_i[oib] < a_i[oia])) {
            a_v = 0.0f;
            b_v = b_x[oib];
            ++oib;
        } else {
            a_v = a_x[oia];
            b_v = b_x[oib];
            ++oia;
            ++oib;
        }
        av.push_back(a_v);
        bv.push_back(b_v);
        if (a_v * b_v < 0.0f) {
            ++n_signflip;
        }
    }

    int k = (int)av.size();
    long long n_inactive = (long long)n_genes - k;
    long long discordant = (long long)n_signflip * n_inactive;

    if (k >= 2) {
        vector<int> idx(k);
        for (int i = 0; i < k; ++i) {
            idx[i] = i;
        }
        sort(idx.begin(), idx.end(), [&](int x, int y) {
            if (av[x] != av[y]) {
                return av[x] < av[y];
            }
            return bv[x] < bv[y];
        });
        vector<float> bs(bv);
        sort(bs.begin(), bs.end());
        bs.erase(unique(bs.begin(), bs.end()), bs.end());
        int B = (int)bs.size();
        vector<int> fen(B + 1, 0);
        auto upd = [&](int p) {
            for (; p <= B; p += p & -p) {
                ++fen[p];
            }
        };
        auto qry = [&](int p) {
            int s = 0;
            for (; p > 0; p -= p & -p) {
                s += fen[p];
            }
            return s;
        };
        auto rankb = [&](float v) {
            return int(lower_bound(bs.begin(), bs.end(), v) - bs.begin()) + 1;
        };
        int inserted = 0;
        int p = 0;
        while (p < k) {
            int q = p;
            while (q < k && av[idx[q]] == av[idx[p]]) {
                ++q;
            }
            for (int t = p; t < q; ++t) {
                int r = rankb(bv[idx[t]]);
                discordant += (long long)(inserted - qry(r));
            }
            for (int t = p; t < q; ++t) {
                upd(rankb(bv[idx[t]]));
                ++inserted;
            }
            p = q;
        }
    }
    return discordant;
}

struct Dataset {
    int n_genes;
    int n_cells;
    vector<vector<float>> dense; // [cell][gene]
    vector<int> csc_p;
    vector<int> csc_i;
    vector<float> csc_x;
};

static Dataset gen(int n_genes, int n_cells, double density, bool signed_vals, mt19937& rng) {
    Dataset ds;
    ds.n_genes = n_genes;
    ds.n_cells = n_cells;
    ds.dense.assign(n_cells, vector<float>(n_genes, 0.0f));
    uniform_real_distribution<double> u(0.0, 1.0);
    uniform_int_distribution<int> mag(1, 10);
    uniform_int_distribution<int> sgn(0, 1);
    ds.csc_p.push_back(0);
    for (int c = 0; c < n_cells; ++c) {
        for (int g = 0; g < n_genes; ++g) {
            if (u(rng) < density) {
                float v = (float)mag(rng);
                if (signed_vals && sgn(rng)) {
                    v = -v;
                }
                ds.dense[c][g] = v;
                ds.csc_i.push_back(g);
                ds.csc_x.push_back(v);
            }
        }
        ds.csc_p.push_back((int)ds.csc_i.size());
    }
    return ds;
}

struct Cfg {
    int n_genes;
    int n_cells;
    double density;
    bool signed_vals;
};

int main() {
    mt19937 rng(12345);

    // The signed configurations exercise the closed form n_signflip * n_inactive.
    vector<Cfg> cfgs = {
        {  50,  20, 0.30, false},
        { 200,  30, 0.10, false},
        { 200,  30, 0.50, false},
        { 500,  25, 0.05, false},
        { 500,  25, 0.30, false},
        {  50,  20, 0.30, true },
        { 200,  30, 0.10, true },
        { 200,  30, 0.50, true },
        { 500,  25, 0.30, true },
        {1000,  20, 0.02, true },
    };

    long long total_pairs = 0;
    long long mismatches = 0;
    int shown = 0;

    for (size_t ci = 0; ci < cfgs.size(); ++ci) {
        const Cfg& cf = cfgs[ci];
        Dataset ds = gen(cf.n_genes, cf.n_cells, cf.density, cf.signed_vals, rng);
        long long cfg_pairs = 0;
        long long cfg_mis = 0;

        for (int a = 0; a < ds.n_cells; ++a) {
            for (int b = a + 1; b < ds.n_cells; ++b) {
                long long d_dense = disc_dense(ds.dense[a], ds.dense[b]);
                long long d_brute = disc_sparse_brute(
                    ds.csc_i.data(), ds.csc_x.data(), ds.csc_p[a], ds.csc_p[a + 1],
                    ds.csc_i.data(), ds.csc_x.data(), ds.csc_p[b], ds.csc_p[b + 1],
                    ds.n_genes
                );
                long long d_fen = disc_sparse_fenwick(
                    ds.csc_i.data(), ds.csc_x.data(), ds.csc_p[a], ds.csc_p[a + 1],
                    ds.csc_i.data(), ds.csc_x.data(), ds.csc_p[b], ds.csc_p[b + 1],
                    ds.n_genes
                );

                ++cfg_pairs;
                bool ok = (d_dense == d_brute) && (d_brute == d_fen);
                if (!ok) {
                    ++cfg_mis;
                    if (shown < 10) {
                        printf(
                            "  MISMATCH cfg#%zu pair(%d,%d): dense=%lld brute=%lld fenwick=%lld\n",
                            ci, a, b, d_dense, d_brute, d_fen
                        );
                        ++shown;
                    }
                }
            }
        }
        total_pairs += cfg_pairs;
        mismatches += cfg_mis;
        const char* sign_label = "nonneg";
        if (cf.signed_vals) {
            sign_label = "signed";
        }
        const char* status_label = "FAIL";
        if (cfg_mis == 0) {
            status_label = "OK";
        }
        printf(
            "cfg#%zu  %5d genes x %3d cells  density=%.2f  %-8s  pairs=%-6lld  %s\n",
            ci, cf.n_genes, cf.n_cells, cf.density, sign_label, cfg_pairs, status_label
        );
    }

    printf("\n==== %lld pairs total, %lld mismatches ====\n", total_pairs, mismatches);

    // Rough timing: k^2 against k log k on a larger sparse case.
    {
        Dataset ds = gen(20000, 60, 0.10, false, rng);
        auto t0 = chrono::steady_clock::now();
        volatile long long acc1 = 0;
        for (int a = 0; a < ds.n_cells; ++a) {
            for (int b = a + 1; b < ds.n_cells; ++b) {
                acc1 += disc_sparse_brute(
                    ds.csc_i.data(), ds.csc_x.data(), ds.csc_p[a], ds.csc_p[a + 1],
                    ds.csc_i.data(), ds.csc_x.data(), ds.csc_p[b], ds.csc_p[b + 1],
                    ds.n_genes
                );
            }
        }
        auto t1 = chrono::steady_clock::now();
        volatile long long acc2 = 0;
        for (int a = 0; a < ds.n_cells; ++a) {
            for (int b = a + 1; b < ds.n_cells; ++b) {
                acc2 += disc_sparse_fenwick(
                    ds.csc_i.data(), ds.csc_x.data(), ds.csc_p[a], ds.csc_p[a + 1],
                    ds.csc_i.data(), ds.csc_x.data(), ds.csc_p[b], ds.csc_p[b + 1],
                    ds.n_genes
                );
            }
        }
        auto t2 = chrono::steady_clock::now();
        double ms_brute = chrono::duration<double, milli>(t1 - t0).count();
        double ms_fen = chrono::duration<double, milli>(t2 - t1).count();
        int nnz_per_cell = (int)(ds.csc_i.size() / ds.n_cells);
        printf("\nTiming (20000 genes x 60 cells, density 0.10, ~%d nnz/cell):\n", nnz_per_cell);
        printf("  brute  O(k^2)    : %8.1f ms\n", ms_brute);
        printf("  fenwick O(k logk): %8.1f ms   (speedup x%.1f)\n", ms_fen, ms_brute / ms_fen);
    }

    if (mismatches == 0) {
        return 0;
    }
    return 1;
}
