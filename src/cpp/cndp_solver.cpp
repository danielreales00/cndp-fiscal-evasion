/**
 * Critical Node Detection Problem (CNDP) Solver
 * C++ implementation for maximum performance.
 *
 * Algorithms:
 *   - ILP exact (Arulselvan et al. 2009) via HiGHS
 *   - Degree heuristic
 *   - Betweenness centrality heuristic (Brandes algorithm)
 *   - Greedy pair-reduction heuristic
 *   - Multi-Start Iterated Local Search (MS-ILS)
 *
 * Build: make (see Makefile)
 * Usage: ./cndp_solver <instance_file> <k> <algorithm> [options]
 */

#include <iostream>
#include <vector>
#include <unordered_set>
#include <algorithm>
#include <numeric>
#include <random>
#include <chrono>
#include <string>
#include <cstdint>
#include <cmath>
#include <queue>

#include "graph.h"
#include "union_find.h"

#ifdef USE_HIGHS
#include "Highs.h"
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

// ============================================================================
// Utility: Timing
// ============================================================================

using Clock = std::chrono::high_resolution_clock;

double elapsed_seconds(Clock::time_point start) {
    auto end = Clock::now();
    return std::chrono::duration<double>(end - start).count();
}

// ============================================================================
// Evaluation: Count connected pairs using Union-Find
// ============================================================================

/**
 * Evaluate the CNDP objective: number of connected pairs in residual graph.
 * The residual graph is G with nodes in `deleted` removed.
 *
 * Uses Union-Find for O(n + m * alpha(n)) evaluation.
 */
int64_t evaluate(const Graph& g, const std::vector<bool>& is_deleted) {
    UnionFind uf(g.n);

    // Process all edges; skip if either endpoint is deleted
    for (auto& [u, v] : g.edges) {
        if (!is_deleted[u] && !is_deleted[v]) {
            uf.unite(u, v);
        }
    }

    // Count pairs, excluding deleted nodes from component sizes
    // Since deleted nodes are isolated (size 1) in UF, we need to subtract them
    int64_t total = 0;
    for (int i = 0; i < g.n; i++) {
        if (!is_deleted[i] && uf.find(i) == i) {
            int64_t s = uf.sz[i];
            total += s * (s - 1) / 2;
        }
    }
    // NOTE: deleted nodes are isolated singletons in UF but we skip them
    // Actually the UF still unions correctly since we skip edges to deleted nodes.
    // But a deleted node might still be a root with sz=1.
    // The loop above skips deleted nodes, so we're fine. But sz might include deleted nodes...
    // No, because we never unite deleted nodes. Each deleted node stays as its own component (sz=1).
    // And we skip them in the pair counting. So the count is correct.

    return total;
}

/**
 * Overload taking a set of deleted node indices.
 */
int64_t evaluate(const Graph& g, const std::unordered_set<int>& deleted) {
    std::vector<bool> is_del(g.n, false);
    for (int v : deleted) is_del[v] = true;
    return evaluate(g, is_del);
}

// ============================================================================
// Greedy Degree Heuristic
// ============================================================================

struct SolverResult {
    std::vector<int> deleted_nodes;
    int64_t objective;
    double time_seconds;
};

SolverResult greedy_degree(const Graph& g, int k) {
    auto start = Clock::now();

    // Compute degrees
    std::vector<std::pair<int, int>> deg_node(g.n);
    for (int i = 0; i < g.n; i++) {
        deg_node[i] = {(int)g.adj[i].size(), i};
    }

    // Sort by degree descending
    std::sort(deg_node.begin(), deg_node.end(), std::greater<>());

    std::vector<bool> is_deleted(g.n, false);
    std::vector<int> deleted;
    for (int i = 0; i < std::min(k, g.n); i++) {
        int node = deg_node[i].second;
        is_deleted[node] = true;
        deleted.push_back(node);
    }

    int64_t obj = evaluate(g, is_deleted);
    double t = elapsed_seconds(start);

    return {deleted, obj, t};
}

// ============================================================================
// Betweenness Centrality (Brandes Algorithm) + Greedy
// ============================================================================

/**
 * Brandes algorithm for betweenness centrality.
 * O(n * m) for unweighted graphs.
 */
std::vector<double> brandes_betweenness(const Graph& g) {
    int n = g.n;
    std::vector<double> bc(n, 0.0);

    for (int s = 0; s < n; s++) {
        std::vector<std::vector<int>> pred(n);
        std::vector<int> sigma(n, 0);
        std::vector<int> dist(n, -1);
        std::vector<double> delta(n, 0.0);

        sigma[s] = 1;
        dist[s] = 0;

        std::queue<int> Q;
        std::vector<int> S; // stack (order of discovery)
        Q.push(s);

        while (!Q.empty()) {
            int v = Q.front(); Q.pop();
            S.push_back(v);

            for (int w : g.adj[v]) {
                if (dist[w] < 0) {
                    dist[w] = dist[v] + 1;
                    Q.push(w);
                }
                if (dist[w] == dist[v] + 1) {
                    sigma[w] += sigma[v];
                    pred[w].push_back(v);
                }
            }
        }

        // Back-propagation
        for (int i = (int)S.size() - 1; i >= 0; i--) {
            int w = S[i];
            for (int v : pred[w]) {
                delta[v] += ((double)sigma[v] / sigma[w]) * (1.0 + delta[w]);
            }
            if (w != s) {
                bc[w] += delta[w];
            }
        }
    }

    // Normalize for undirected graph
    for (int i = 0; i < n; i++) {
        bc[i] /= 2.0;
    }

    return bc;
}

SolverResult greedy_betweenness(const Graph& g, int k) {
    auto start = Clock::now();

    std::vector<double> bc = brandes_betweenness(g);

    // Sort by betweenness descending
    std::vector<std::pair<double, int>> bc_node(g.n);
    for (int i = 0; i < g.n; i++) {
        bc_node[i] = {bc[i], i};
    }
    std::sort(bc_node.begin(), bc_node.end(), std::greater<>());

    std::vector<bool> is_deleted(g.n, false);
    std::vector<int> deleted;
    for (int i = 0; i < std::min(k, g.n); i++) {
        int node = bc_node[i].second;
        is_deleted[node] = true;
        deleted.push_back(node);
    }

    int64_t obj = evaluate(g, is_deleted);
    double t = elapsed_seconds(start);

    return {deleted, obj, t};
}

// ============================================================================
// Greedy Pair-Reduction Heuristic
// ============================================================================

/**
 * Compute the reduction in connected pairs if we remove `node` from the
 * residual graph (with `is_deleted` already removed).
 *
 * Strategy: The component containing `node` will split into sub-components.
 * Reduction = C(comp_size, 2) - sum(C(sub_size, 2))
 */
int64_t compute_pair_reduction(const Graph& g, int node, const std::vector<bool>& is_deleted) {
    // BFS to find the component containing `node`
    std::vector<bool> in_comp(g.n, false);
    std::queue<int> bfs;
    bfs.push(node);
    in_comp[node] = true;
    int comp_size = 0;

    while (!bfs.empty()) {
        int v = bfs.front(); bfs.pop();
        comp_size++;
        for (int nb : g.adj[v]) {
            if (!is_deleted[nb] && !in_comp[nb]) {
                in_comp[nb] = true;
                bfs.push(nb);
            }
        }
    }

    int64_t before_pairs = (int64_t)comp_size * (comp_size - 1) / 2;

    // BFS from neighbors of `node` to find sub-components after removal
    std::vector<bool> visited(g.n, false);
    visited[node] = true;
    int64_t after_pairs = 0;

    for (int nb : g.adj[node]) {
        if (is_deleted[nb] || visited[nb]) continue;
        // BFS from nb
        std::queue<int> bfs2;
        bfs2.push(nb);
        visited[nb] = true;
        int sub_size = 0;

        while (!bfs2.empty()) {
            int v = bfs2.front(); bfs2.pop();
            sub_size++;
            for (int w : g.adj[v]) {
                if (!is_deleted[w] && !visited[w] && w != node) {
                    visited[w] = true;
                    bfs2.push(w);
                }
            }
        }
        after_pairs += (int64_t)sub_size * (sub_size - 1) / 2;
    }

    // Also count nodes in the component that are not reachable from any neighbor
    // (shouldn't happen in a connected component, but handle gracefully)
    // Actually all nodes in the component are either `node` itself or reachable from a neighbor.

    return before_pairs - after_pairs;
}

SolverResult greedy_pair_reduction(const Graph& g, int k) {
    auto start = Clock::now();

    std::vector<bool> is_deleted(g.n, false);
    std::vector<int> deleted;

    for (int round = 0; round < k; round++) {
        int best_node = -1;
        int64_t best_reduction = -1;

        for (int v = 0; v < g.n; v++) {
            if (is_deleted[v]) continue;
            // Check if node has any non-deleted neighbors (degree > 0 in residual)
            bool has_neighbor = false;
            for (int nb : g.adj[v]) {
                if (!is_deleted[nb]) { has_neighbor = true; break; }
            }

            if (!has_neighbor) {
                // Isolated node - removal reduces 0 pairs
                if (best_node < 0) best_node = v; // fallback
                continue;
            }

            int64_t reduction = compute_pair_reduction(g, v, is_deleted);
            if (reduction > best_reduction) {
                best_reduction = reduction;
                best_node = v;
            }
        }

        if (best_node < 0) break;
        is_deleted[best_node] = true;
        deleted.push_back(best_node);
    }

    int64_t obj = evaluate(g, is_deleted);
    double t = elapsed_seconds(start);

    return {deleted, obj, t};
}

// ============================================================================
// Local Search (Swap Neighborhood, First-Improvement)
// ============================================================================

SolverResult local_search(const Graph& g, int k,
                          const std::vector<int>& initial_deleted,
                          int max_iterations, std::mt19937& rng) {
    auto start = Clock::now();

    std::unordered_set<int> deleted_set(initial_deleted.begin(), initial_deleted.end());
    std::vector<bool> is_deleted(g.n, false);
    for (int v : initial_deleted) is_deleted[v] = true;

    int64_t current_obj = evaluate(g, is_deleted);

    for (int iteration = 0; iteration < max_iterations; iteration++) {
        bool improved = false;

        // Shuffle deleted list
        std::vector<int> del_list(deleted_set.begin(), deleted_set.end());
        std::shuffle(del_list.begin(), del_list.end(), rng);

        for (int node_out : del_list) {
            // Build list of non-deleted nodes
            std::vector<int> non_deleted;
            non_deleted.reserve(g.n - k);
            for (int v = 0; v < g.n; v++) {
                if (!is_deleted[v]) non_deleted.push_back(v);
            }
            std::shuffle(non_deleted.begin(), non_deleted.end(), rng);

            bool found_swap = false;
            for (int node_in : non_deleted) {
                // Evaluate swap: remove node_out from deleted, add node_in
                is_deleted[node_out] = false;
                is_deleted[node_in] = true;

                int64_t new_obj = evaluate(g, is_deleted);

                if (new_obj < current_obj) {
                    // Accept swap (first-improvement)
                    deleted_set.erase(node_out);
                    deleted_set.insert(node_in);
                    current_obj = new_obj;
                    found_swap = true;
                    break; // first-improvement
                } else {
                    // Revert
                    is_deleted[node_out] = true;
                    is_deleted[node_in] = false;
                }
            }

            if (found_swap) {
                improved = true;
                break; // restart scan
            }
        }

        if (!improved) break;
    }

    std::vector<int> result(deleted_set.begin(), deleted_set.end());
    double t = elapsed_seconds(start);
    return {result, current_obj, t};
}

// ============================================================================
// Perturbation
// ============================================================================

std::vector<int> perturb(const Graph& g, const std::vector<int>& deleted,
                         int strength, std::mt19937& rng) {
    std::unordered_set<int> new_deleted(deleted.begin(), deleted.end());
    std::vector<int> non_deleted;
    non_deleted.reserve(g.n);
    std::vector<bool> is_del(g.n, false);
    for (int v : deleted) is_del[v] = true;
    for (int v = 0; v < g.n; v++) {
        if (!is_del[v]) non_deleted.push_back(v);
    }

    int num_swaps = std::min({strength, (int)new_deleted.size(), (int)non_deleted.size()});

    // Select random nodes to swap out
    std::vector<int> del_vec(new_deleted.begin(), new_deleted.end());
    std::shuffle(del_vec.begin(), del_vec.end(), rng);
    std::shuffle(non_deleted.begin(), non_deleted.end(), rng);

    for (int i = 0; i < num_swaps; i++) {
        new_deleted.erase(del_vec[i]);
        new_deleted.insert(non_deleted[i]);
    }

    return std::vector<int>(new_deleted.begin(), new_deleted.end());
}

// ============================================================================
// Iterated Local Search (ILS)
// ============================================================================

SolverResult ils(const Graph& g, int k, const std::vector<int>& initial_deleted,
                 int max_restarts, int ls_iterations, int perturbation_strength,
                 int seed) {
    auto start = Clock::now();
    std::mt19937 rng(seed);

    // Initial local search
    SolverResult ls_result = local_search(g, k, initial_deleted, ls_iterations, rng);
    std::vector<int> best_deleted = ls_result.deleted_nodes;
    int64_t best_obj = ls_result.objective;
    std::vector<int> current_deleted = best_deleted;

    for (int r = 0; r < max_restarts; r++) {
        // Perturb
        std::vector<int> perturbed = perturb(g, current_deleted, perturbation_strength, rng);

        // Local search from perturbed solution
        SolverResult new_result = local_search(g, k, perturbed, ls_iterations, rng);
        std::vector<int> new_deleted = new_result.deleted_nodes;
        int64_t new_obj = new_result.objective;

        if (new_obj < best_obj) {
            best_obj = new_obj;
            best_deleted = new_deleted;
            current_deleted = new_deleted;
        } else {
            // With probability 0.3, accept the new solution anyway (diversification)
            std::uniform_real_distribution<double> dist(0.0, 1.0);
            if (dist(rng) < 0.3) {
                current_deleted = new_deleted;
            } else {
                current_deleted = best_deleted;
            }
        }
    }

    double t = elapsed_seconds(start);
    return {best_deleted, best_obj, t};
}

// ============================================================================
// Multi-Start ILS (with OpenMP parallelism)
// ============================================================================

SolverResult multi_start_ils(const Graph& g, int k, int n_restarts, int n_threads,
                             int ls_iterations, int perturbation_strength, int seed) {
    auto start = Clock::now();
    std::mt19937 rng(seed);

    // Phase 1: Generate diverse initial solutions
    struct StartSolution {
        std::vector<int> deleted;
        int64_t obj;
        std::string name;
    };
    std::vector<StartSolution> starts;

    // Degree heuristic
    auto deg_result = greedy_degree(g, k);
    starts.push_back({deg_result.deleted_nodes, deg_result.objective, "degree"});

    // Betweenness heuristic
    auto bet_result = greedy_betweenness(g, k);
    starts.push_back({bet_result.deleted_nodes, bet_result.objective, "betweenness"});

    // Greedy pair-reduction
    auto gr_result = greedy_pair_reduction(g, k);
    starts.push_back({gr_result.deleted_nodes, gr_result.objective, "greedy"});

    // Random starts
    std::vector<int> all_nodes(g.n);
    std::iota(all_nodes.begin(), all_nodes.end(), 0);
    int n_random = std::max(1, n_threads - 3);
    for (int i = 0; i < n_random; i++) {
        std::shuffle(all_nodes.begin(), all_nodes.end(), rng);
        std::vector<int> rand_del(all_nodes.begin(), all_nodes.begin() + std::min(k, g.n));
        int64_t rand_obj = evaluate(g, std::unordered_set<int>(rand_del.begin(), rand_del.end()));
        starts.push_back({rand_del, rand_obj, "random_" + std::to_string(i)});
    }

    // Sort by objective (best first)
    std::sort(starts.begin(), starts.end(),
              [](const StartSolution& a, const StartSolution& b) {
                  return a.obj < b.obj;
              });

    int n_workers = std::min(n_threads, (int)starts.size());
    int restarts_per_start = std::max(1, n_restarts / n_workers);

    // Phase 2: Run ILS from best starting points in parallel
    std::vector<SolverResult> results(n_workers);

#ifdef _OPENMP
    #pragma omp parallel for num_threads(n_threads) schedule(dynamic)
#endif
    for (int i = 0; i < n_workers; i++) {
        int worker_seed = seed + (int)(std::hash<std::string>{}(starts[i].name) % 10000);
        results[i] = ils(g, k, starts[i].deleted, restarts_per_start,
                         ls_iterations, perturbation_strength, worker_seed);
    }

    // Phase 3: Find overall best
    int64_t best_obj = starts[0].obj;
    std::vector<int> best_deleted = starts[0].deleted;

    for (int i = 0; i < n_workers; i++) {
        if (results[i].objective < best_obj) {
            best_obj = results[i].objective;
            best_deleted = results[i].deleted_nodes;
        }
    }

    double t = elapsed_seconds(start);
    return {best_deleted, best_obj, t};
}

// ============================================================================
// ILP Exact Solver (Arulselvan et al. 2009) via HiGHS
// ============================================================================

#ifdef USE_HIGHS
SolverResult solve_ilp(const Graph& g, int k, int time_limit, int threads) {
    auto start = Clock::now();

    int n = g.n;

    // For very large graphs, the O(n^3) transitivity constraints are infeasible.
    // Warn and bail.
    if (n > 200) {
        std::cerr << "WARNING: ILP with transitivity constraints is O(n^3). "
                  << "n=" << n << " may be too large. Proceeding anyway..." << std::endl;
    }

    // Build model using HighsModel/HighsLp directly for better API compatibility
    HighsModel model;
    HighsLp& lp = model.lp_;

    int n_vars = n + n * (n - 1) / 2;

    // Helper to get u_ij variable index (i < j)
    auto u_idx = [&](int i, int j) -> int {
        return n + (int)((int64_t)i * (2 * n - i - 1) / 2 + (j - i - 1));
    };

    lp.num_col_ = n_vars;
    lp.num_row_ = 0;
    lp.sense_ = ObjSense::kMinimize;

    // Column bounds and costs
    lp.col_cost_.resize(n_vars, 0.0);
    lp.col_lower_.resize(n_vars, 0.0);
    lp.col_upper_.resize(n_vars, 1.0);

    // Objective: minimize sum u_ij
    for (int i = n; i < n_vars; i++) {
        lp.col_cost_[i] = 1.0;
    }

    // Integrality: x_i are integer (binary)
    lp.integrality_.resize(n_vars, HighsVarType::kContinuous);
    for (int i = 0; i < n; i++) {
        lp.integrality_[i] = HighsVarType::kInteger;
    }

    // Build constraints using CSC format (compressed sparse column)
    // We'll build row-wise first, then convert
    struct RowEntry {
        double lower;
        double upper;
        std::vector<std::pair<int, double>> coeffs;
    };
    std::vector<RowEntry> rows;

    // Constraint 1: Budget - sum x_i <= k
    {
        RowEntry row;
        row.lower = -1e30;
        row.upper = (double)k;
        for (int i = 0; i < n; i++) {
            row.coeffs.push_back({i, 1.0});
        }
        rows.push_back(std::move(row));
    }

    // Constraint 2: Edge constraints - u_ij + x_i + x_j >= 1 for (i,j) in E
    for (auto& edge : g.edges) {
        int i = std::min(edge.first, edge.second);
        int j = std::max(edge.first, edge.second);
        RowEntry row;
        row.lower = 1.0;
        row.upper = 1e30;
        row.coeffs.push_back({u_idx(i, j), 1.0});
        row.coeffs.push_back({i, 1.0});
        row.coeffs.push_back({j, 1.0});
        rows.push_back(std::move(row));
    }

    // Constraint 3: Disconnection - u_ij + x_i <= 1 and u_ij + x_j <= 1
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            int uij = u_idx(i, j);
            {
                RowEntry row;
                row.lower = -1e30;
                row.upper = 1.0;
                row.coeffs.push_back({uij, 1.0});
                row.coeffs.push_back({i, 1.0});
                rows.push_back(std::move(row));
            }
            {
                RowEntry row;
                row.lower = -1e30;
                row.upper = 1.0;
                row.coeffs.push_back({uij, 1.0});
                row.coeffs.push_back({j, 1.0});
                rows.push_back(std::move(row));
            }
        }
    }

    // Constraint 4: Transitivity - u_ij - u_ik - u_jk >= -1
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            for (int kk = 0; kk < n; kk++) {
                if (kk == i || kk == j) continue;
                int uij = u_idx(i, j);
                int uik = (i < kk) ? u_idx(i, kk) : u_idx(kk, i);
                int ujk = (j < kk) ? u_idx(j, kk) : u_idx(kk, j);
                RowEntry row;
                row.lower = -1.0;
                row.upper = 1e30;
                row.coeffs.push_back({uij, 1.0});
                row.coeffs.push_back({uik, -1.0});
                row.coeffs.push_back({ujk, -1.0});
                rows.push_back(std::move(row));
            }
        }
    }

    // Convert to CSC (HiGHS format): a_matrix_ in column-wise format
    int num_rows = (int)rows.size();
    lp.num_row_ = num_rows;
    lp.row_lower_.resize(num_rows);
    lp.row_upper_.resize(num_rows);

    // Build in row-wise format
    lp.a_matrix_.format_ = MatrixFormat::kRowwise;
    lp.a_matrix_.num_col_ = n_vars;
    lp.a_matrix_.num_row_ = num_rows;
    lp.a_matrix_.start_.resize(num_rows + 1);
    lp.a_matrix_.start_[0] = 0;

    for (int r = 0; r < num_rows; r++) {
        lp.row_lower_[r] = rows[r].lower;
        lp.row_upper_[r] = rows[r].upper;
        for (size_t c = 0; c < rows[r].coeffs.size(); c++) {
            lp.a_matrix_.index_.push_back(rows[r].coeffs[c].first);
            lp.a_matrix_.value_.push_back(rows[r].coeffs[c].second);
        }
        lp.a_matrix_.start_[r + 1] = (HighsInt)lp.a_matrix_.index_.size();
    }

    // Create HiGHS instance and solve
    Highs highs;
    highs.setOptionValue("output_flag", false);
    highs.setOptionValue("threads", threads);
    highs.setOptionValue("time_limit", (double)time_limit);
    highs.setOptionValue("mip_rel_gap", 0.0001);

    highs.passModel(model);
    highs.run();

    // Extract solution
    std::vector<int> deleted;
    int64_t obj = -1;

    HighsModelStatus status = highs.getModelStatus();
    // Check if we have a feasible solution (optimal, or feasible at time/solution limit)
    bool has_solution = (status == HighsModelStatus::kOptimal ||
                         status == HighsModelStatus::kObjectiveBound ||
                         status == HighsModelStatus::kSolutionLimit);

    // For time limit, check if there's a feasible incumbent
    if (!has_solution && status == HighsModelStatus::kTimeLimit) {
        // HiGHS stores incumbent; try to read it
        double obj_val = 0.0;
        if (highs.getInfoValue("objective_function_value", obj_val) == HighsStatus::kOk) {
            has_solution = (obj_val < 1e29);
        }
    }

    if (has_solution) {
        const HighsSolution& sol = highs.getSolution();
        for (int i = 0; i < n; i++) {
            if (sol.col_value[i] > 0.5) {
                deleted.push_back(i);
            }
        }
        double obj_val = 0.0;
        highs.getInfoValue("objective_function_value", obj_val);
        obj = (int64_t)std::round(obj_val);
    }

    double t = elapsed_seconds(start);
    return {deleted, obj, t};
}
#else
SolverResult solve_ilp(const Graph& g, int k, int time_limit, int threads) {
    (void)g; (void)k; (void)time_limit; (void)threads;
    std::cerr << "ERROR: ILP solver not available. Compile with USE_HIGHS=1." << std::endl;
    return {{}, -1, 0.0};
}
#endif

// ============================================================================
// JSON Output
// ============================================================================

void print_json_result(const std::string& instance, const Graph& g, int k,
                       int64_t initial_pairs, const std::string& algorithm,
                       const SolverResult& result) {
    std::cout << "{\n";
    std::cout << "  \"instance\": \"" << instance << "\",\n";
    std::cout << "  \"n\": " << g.n << ", \"m\": " << g.m << ", \"k\": " << k << ",\n";
    std::cout << "  \"initial_pairs\": " << initial_pairs << ",\n";
    std::cout << "  \"algorithm\": \"" << algorithm << "\",\n";
    std::cout << "  \"objective\": " << result.objective << ",\n";
    std::cout << "  \"time_seconds\": " << result.time_seconds << ",\n";
    std::cout << "  \"deleted_nodes\": [";
    std::vector<int> sorted_del = result.deleted_nodes;
    std::sort(sorted_del.begin(), sorted_del.end());
    for (size_t i = 0; i < sorted_del.size(); i++) {
        if (i > 0) std::cout << ", ";
        std::cout << sorted_del[i];
    }
    std::cout << "]\n";
    std::cout << "}\n";
}

// ============================================================================
// Main: CLI Interface
// ============================================================================

void print_usage() {
    std::cerr << "Usage: ./cndp_solver <instance_file> <k> <algorithm> [options]\n\n";
    std::cerr << "Algorithms:\n";
    std::cerr << "  ilp          - ILP exact (Arulselvan 2009)\n";
    std::cerr << "  degree       - Degree heuristic\n";
    std::cerr << "  betweenness  - Betweenness heuristic\n";
    std::cerr << "  greedy       - Greedy pair-reduction\n";
    std::cerr << "  ms_ils       - Multi-Start ILS (our algorithm)\n";
    std::cerr << "  all          - Run all algorithms\n\n";
    std::cerr << "Options:\n";
    std::cerr << "  --time-limit <seconds>   (for ILP, default 300)\n";
    std::cerr << "  --restarts <n>           (for ILS, default 25)\n";
    std::cerr << "  --threads <n>            (default 8)\n";
    std::cerr << "  --ls-iter <n>            (local search iterations, default 50)\n";
    std::cerr << "  --perturb <n>            (perturbation strength, default 3)\n";
    std::cerr << "  --seed <n>               (random seed, default 42)\n";
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        print_usage();
        return 1;
    }

    std::string instance_file = argv[1];
    int k = std::stoi(argv[2]);
    std::string algorithm = argv[3];

    // Parse options
    int time_limit = 300;
    int restarts = 25;
    int threads = 8;
    int ls_iter = 50;
    int perturb_strength = 3;
    int seed = 42;

    for (int i = 4; i < argc; i++) {
        std::string opt = argv[i];
        if (opt == "--time-limit" && i + 1 < argc) {
            time_limit = std::stoi(argv[++i]);
        } else if (opt == "--restarts" && i + 1 < argc) {
            restarts = std::stoi(argv[++i]);
        } else if (opt == "--threads" && i + 1 < argc) {
            threads = std::stoi(argv[++i]);
        } else if (opt == "--ls-iter" && i + 1 < argc) {
            ls_iter = std::stoi(argv[++i]);
        } else if (opt == "--perturb" && i + 1 < argc) {
            perturb_strength = std::stoi(argv[++i]);
        } else if (opt == "--seed" && i + 1 < argc) {
            seed = std::stoi(argv[++i]);
        } else {
            std::cerr << "Unknown option: " << opt << std::endl;
            print_usage();
            return 1;
        }
    }

    // Load graph
    Graph g;
    try {
        g = Graph::load(instance_file);
    } catch (const std::exception& e) {
        std::cerr << "Error loading graph: " << e.what() << std::endl;
        return 1;
    }

    // Compute initial pairs (objective with no deletions)
    std::vector<bool> no_del(g.n, false);
    int64_t initial_pairs = evaluate(g, no_del);

    // Extract instance name from path
    std::string instance_name = instance_file;
    size_t slash_pos = instance_name.rfind('/');
    if (slash_pos != std::string::npos) {
        instance_name = instance_name.substr(slash_pos + 1);
    }

    // Run algorithm(s)
    auto run_algorithm = [&](const std::string& algo) {
        SolverResult result;
        if (algo == "ilp") {
            result = solve_ilp(g, k, time_limit, threads);
        } else if (algo == "degree") {
            result = greedy_degree(g, k);
        } else if (algo == "betweenness") {
            result = greedy_betweenness(g, k);
        } else if (algo == "greedy") {
            result = greedy_pair_reduction(g, k);
        } else if (algo == "ms_ils") {
            result = multi_start_ils(g, k, restarts, threads, ls_iter, perturb_strength, seed);
        } else {
            std::cerr << "Unknown algorithm: " << algo << std::endl;
            return;
        }
        print_json_result(instance_name, g, k, initial_pairs, algo, result);
    };

    if (algorithm == "all") {
        std::vector<std::string> algos = {"degree", "betweenness", "greedy", "ms_ils", "ilp"};
        std::cout << "[\n";
        for (size_t i = 0; i < algos.size(); i++) {
            if (i > 0) std::cout << ",\n";
            run_algorithm(algos[i]);
        }
        std::cout << "]\n";
    } else {
        run_algorithm(algorithm);
    }

    return 0;
}
