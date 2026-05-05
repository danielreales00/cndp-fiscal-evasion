#ifndef GRAPH_H
#define GRAPH_H

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>

/**
 * Simple undirected graph using adjacency list representation.
 * Nodes are always 0-indexed internally.
 */
struct Graph {
    int n;  // number of nodes
    int m;  // number of edges
    std::vector<std::vector<int>> adj;

    // Edge list for fast iteration over all edges
    std::vector<std::pair<int, int>> edges;

    Graph() : n(0), m(0) {}

    explicit Graph(int num_nodes) : n(num_nodes), m(0), adj(num_nodes) {}

    void add_edge(int u, int v) {
        if (u == v) return; // no self-loops
        adj[u].push_back(v);
        adj[v].push_back(u);
        if (u < v)
            edges.emplace_back(u, v);
        else
            edges.emplace_back(v, u);
        m++;
    }

    /**
     * Load from METIS format (.graph files).
     * First line: "n m [fmt]"
     * Then n lines, each listing neighbors (1-indexed).
     * We convert to 0-indexed internally.
     */
    static Graph load_metis(const std::string& filepath) {
        std::ifstream file(filepath);
        if (!file.is_open())
            throw std::runtime_error("Cannot open file: " + filepath);

        std::string line;
        // Skip comment lines starting with %
        while (std::getline(file, line)) {
            if (!line.empty() && line[0] != '%') break;
        }

        std::istringstream header(line);
        int n_nodes = 0, n_edges = 0;
        header >> n_nodes >> n_edges;
        (void)n_edges; // from METIS header, not used directly

        Graph g(n_nodes);

        for (int i = 0; i < n_nodes; i++) {
            if (!std::getline(file, line)) break;
            // Skip comment lines
            if (!line.empty() && line[0] == '%') { i--; continue; }
            std::istringstream iss(line);
            int nb;
            while (iss >> nb) {
                nb--; // convert to 0-indexed
                if (nb > i) {
                    g.add_edge(i, nb);
                }
            }
        }

        return g;
    }

    /**
     * Load from IRMS adjacency-list format.
     * First line: N (number of nodes)
     * Then: "vertex: neighbor1 neighbor2 ..." (0-indexed)
     */
    static Graph load_irms(const std::string& filepath) {
        std::ifstream file(filepath);
        if (!file.is_open())
            throw std::runtime_error("Cannot open file: " + filepath);

        std::string line;
        std::getline(file, line);
        int n_nodes = std::stoi(line);
        Graph g(n_nodes);

        while (std::getline(file, line)) {
            if (line.empty()) continue;
            size_t colon = line.find(':');
            if (colon == std::string::npos) continue;
            int u = std::stoi(line.substr(0, colon));
            std::istringstream iss(line.substr(colon + 1));
            int v;
            while (iss >> v) {
                if (v > u) {
                    g.add_edge(u, v);
                }
            }
        }
        return g;
    }

    /**
     * Load from edge-list format (.txt files).
     * Lines starting with # or containing "nodes" are headers/comments.
     * Remaining lines: "u v" pairs (0-indexed).
     * Auto-detects IRMS adjacency-list format (lines with ':').
     */
    static Graph load_edgelist(const std::string& filepath) {
        std::ifstream file(filepath);
        if (!file.is_open())
            throw std::runtime_error("Cannot open file: " + filepath);

        // Peek to detect format: if second non-empty line contains ':', it's IRMS format
        std::string first_line, second_line;
        auto start_pos = file.tellg();
        while (std::getline(file, first_line)) {
            if (!first_line.empty() && first_line[0] != '#') break;
        }
        while (std::getline(file, second_line)) {
            if (!second_line.empty() && second_line[0] != '#') break;
        }
        file.clear();
        file.seekg(start_pos);

        if (second_line.find(':') != std::string::npos) {
            file.close();
            return load_irms(filepath);
        }

        // Standard edge-list format
        std::vector<std::pair<int, int>> edge_pairs;
        std::string line;
        int max_node = -1;

        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::string lower_line = line;
            std::transform(lower_line.begin(), lower_line.end(), lower_line.begin(), ::tolower);
            if (lower_line.find("nodes") != std::string::npos) continue;

            std::istringstream iss(line);
            int u, v;
            if (iss >> u >> v) {
                edge_pairs.emplace_back(u, v);
                max_node = std::max(max_node, std::max(u, v));
            }
        }

        Graph g(max_node + 1);
        for (auto& [u, v] : edge_pairs) {
            if (u != v) {
                g.add_edge(u, v);
            }
        }

        return g;
    }

    /**
     * Auto-detect format by file extension and load.
     */
    static Graph load(const std::string& filepath) {
        size_t dot_pos = filepath.rfind('.');
        if (dot_pos == std::string::npos)
            throw std::runtime_error("Cannot determine file format: " + filepath);

        std::string ext = filepath.substr(dot_pos);
        if (ext == ".graph")
            return load_metis(filepath);
        else if (ext == ".txt")
            return load_edgelist(filepath);
        else
            throw std::runtime_error("Unknown file format: " + ext);
    }
};

#endif // GRAPH_H
