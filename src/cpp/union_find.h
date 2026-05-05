#ifndef UNION_FIND_H
#define UNION_FIND_H

#include <vector>
#include <numeric>
#include <cstdint>

/**
 * Disjoint Set Union (Union-Find) with path compression and union by size.
 * Used for fast component counting in CNDP evaluation.
 */
class UnionFind {
public:
    std::vector<int> parent;
    std::vector<int> sz;
    int n;
    int num_components;

    UnionFind() : n(0), num_components(0) {}

    explicit UnionFind(int n) : n(n), num_components(n) {
        parent.resize(n);
        sz.resize(n, 1);
        std::iota(parent.begin(), parent.end(), 0);
    }

    void reset(int new_n) {
        n = new_n;
        num_components = new_n;
        parent.resize(n);
        sz.resize(n);
        std::iota(parent.begin(), parent.end(), 0);
        std::fill(sz.begin(), sz.end(), 1);
    }

    int find(int x) {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]]; // path compression (two-pass halving)
            x = parent[x];
        }
        return x;
    }

    bool unite(int a, int b) {
        a = find(a);
        b = find(b);
        if (a == b) return false;
        // Union by size
        if (sz[a] < sz[b]) std::swap(a, b);
        parent[b] = a;
        sz[a] += sz[b];
        num_components--;
        return true;
    }

    bool connected(int a, int b) {
        return find(a) == find(b);
    }

    /**
     * Count total connected pairs: sum C(sz[root], 2) over all roots.
     * This is the CNDP objective function value for the residual graph.
     */
    int64_t count_connected_pairs() const {
        int64_t total = 0;
        for (int i = 0; i < n; i++) {
            if (parent[i] == i) {
                int64_t s = sz[i];
                total += s * (s - 1) / 2;
            }
        }
        return total;
    }
};

#endif // UNION_FIND_H
