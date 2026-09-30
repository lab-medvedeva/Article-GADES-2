// Dense Kendall: naive O(n^2) discordant count against the Fenwick-tree
// O(n log n) count used by the dense Kendall kernel of GADES.
// Checks that both give the same number of discordant pairs for every pair of
// objects and reports the running time of each. Standalone: needs neither
// GADES nor a GPU. Exit status is 0 when there are no mismatches.
//
// Build:  g++ -O2 -std=c++17 kendall_dense_reference.cpp -o kendall_dense_reference
// Run:    ./kendall_dense_reference

#include <cstdio>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
using namespace std;

// Naive count of pairs (i < j) with (a_i - a_j) * (b_i - b_j) < 0.
static long long naive(const float* a, const float* b, int n) {
  long long d = 0;
  for (int i = 0; i < n; ++i) {
    for (int j = i + 1; j < n; ++j) {
      if ((a[i] - a[j]) * (b[i] - b[j]) < 0.0f) {
        ++d;
      }
    }
  }
  return d;
}

// Fenwick-tree count, O(n log n). Same algorithm as dense_kendall_disc, the
// dense Kendall kernel of GADES.
static long long fenwick(const float* a, const float* b, int n, int* idx, float* bs, int* fen) {
  if (n < 2) {
    return 0;
  }
  for (int i = 0; i < n; ++i) {
    idx[i] = i;
  }
  std::sort(idx, idx + n, [&](int x, int y) {
    if (a[x] != a[y]) {
      return a[x] < a[y];
    }
    return b[x] < b[y];
  });
  for (int i = 0; i < n; ++i) {
    bs[i] = b[i];
  }
  std::sort(bs, bs + n);
  int B = (int)(std::unique(bs, bs + n) - bs);
  for (int i = 0; i <= B; ++i) {
    fen[i] = 0;
  }
  long long disc = 0;
  int inserted = 0;
  int p = 0;
  while (p < n) {
    int q = p;
    while (q < n && a[idx[q]] == a[idx[p]]) {
      ++q;
    }
    for (int t = p; t < q; ++t) {
      int r = (int)(std::lower_bound(bs, bs + B, b[idx[t]]) - bs) + 1;
      int le = 0;
      for (int i = r; i > 0; i -= i & -i) {
        le += fen[i];
      }
      disc += inserted - le;
    }
    for (int t = p; t < q; ++t) {
      int r = (int)(std::lower_bound(bs, bs + B, b[idx[t]]) - bs) + 1;
      for (int i = r; i <= B; i += i & -i) {
        ++fen[i];
      }
      ++inserted;
    }
    p = q;
  }
  return disc;
}

int main() {
  // features x objects, single-threaded comparison
  int n = 2000;
  int m = 80;
  mt19937 rng(123);
  uniform_int_distribution<int> v(1, 8);
  vector<vector<float>> col(m, vector<float>(n));
  for (int c = 0; c < m; ++c) {
    for (int i = 0; i < n; ++i) {
      col[c][i] = (float)v(rng);
    }
  }

  long long mism = 0;
  long long total = 0;
  vector<int> idx(n);
  vector<int> fen(n + 1);
  vector<float> bs(n);

  auto t0 = chrono::steady_clock::now();
  volatile long long acc1 = 0;
  for (int a = 0; a < m; ++a) {
    for (int b = a + 1; b < m; ++b) {
      acc1 += naive(col[a].data(), col[b].data(), n);
    }
  }
  auto t1 = chrono::steady_clock::now();
  volatile long long acc2 = 0;
  for (int a = 0; a < m; ++a) {
    for (int b = a + 1; b < m; ++b) {
      long long fn = fenwick(col[a].data(), col[b].data(), n, idx.data(), bs.data(), fen.data());
      // naive count recomputed for the exact check
      long long nv = naive(col[a].data(), col[b].data(), n);
      ++total;
      if (fn != nv) {
        ++mism;
      }
      acc2 += fn;
    }
  }
  // Fenwick timing alone, without the naive recount
  auto t2 = chrono::steady_clock::now();
  volatile long long acc3 = 0;
  for (int a = 0; a < m; ++a) {
    for (int b = a + 1; b < m; ++b) {
      acc3 += fenwick(col[a].data(), col[b].data(), n, idx.data(), bs.data(), fen.data());
    }
  }
  auto t3 = chrono::steady_clock::now();

  double ms_naive = chrono::duration<double, milli>(t1 - t0).count();
  double ms_fen = chrono::duration<double, milli>(t3 - t2).count();
  printf("dense %d features x %d cells (%lld cell-pairs), single thread\n", n, m, total);
  printf("  mismatches (naive vs fenwick): %lld\n", mism);
  printf("  naive   O(n^2)    : %9.1f ms\n", ms_naive);
  printf("  fenwick O(n log n): %9.1f ms   (speedup x%.1f)\n", ms_fen, ms_naive / ms_fen);
  if (mism == 0) {
    return 0;
  }
  return 1;
}
