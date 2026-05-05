# Critical Node Detection Problem: ILP, MS-ILS, and IRMS Comparison

**Solving the Critical Node Detection Problem: An ILP Formulation and a Multi-Start Iterated Local Search Heuristic Applied to Fiscal Evasion Networks**

Daniel Reales — Universidad de los Andes, ISIS4208 Análisis de Algoritmos (2026-10)

Juan David Ceballos — Universidad de los Andes, ISIS4208 Análisis de Algoritmos (2026-10)

## Overview

This repository contains the implementation, data, and reproduction scripts for our comparative study of three approaches to the Critical Node Detection Problem (CNDP):

1. **ILP** — Integer Linear Programming formulation (Arulselvan et al., 2009) using HiGHS
2. **MS-ILS** — Multi-Start Iterated Local Search (our algorithm)
3. **IRMS** — Reduce-Solve-Combine Memetic Search (Zhou et al., 2023, state-of-the-art)

We apply CNDP to **fiscal evasion network disruption** using real data from the ICIJ Offshore Leaks database (Panama Papers).

## Repository Structure

```
.
├── paper.tex                     # LaTeX source of the paper
├── paper.pdf                     # Compiled paper
├── src/
│   ├── cpp/
│   │   ├── cndp_solver.cpp       # Main solver: ILP + MS-ILS + heuristics
│   │   ├── IRMS_modified.cpp     # Modified IRMS (Zhou et al. 2023)
│   │   ├── graph.h               # Graph data structure and loaders
│   │   ├── union_find.h          # Union-Find for objective evaluation
│   │   └── Makefile              # Build system (macOS/Linux)
│   ├── run_final_benchmarks.py   # Full benchmark suite (reproduces paper results)
│   ├── run_cpp_benchmarks.py     # Quick/smoke benchmarks for testing
│   └── generate_paper_figures.py # Generate all figures from results JSON
├── data/
│   ├── icij_colombia/            # Colombian fiscal evasion graphs (included)
│   ├── *.graph                   # Small test instances (karate, dolphins, etc.)
│   └── irms_format/              # Converted test instances
├── results/                      # Benchmark results (JSON)
│   ├── benchmark_irms_paper.json # Results on 12 IRMS paper instances
│   └── benchmark_colombia.json   # Results on 4 Colombian instances
└── figures/                      # Generated figures (PDF + PNG)
```

## Requirements

### System
- macOS (Apple Silicon or Intel) or Linux
- C++17 compiler (clang++ or g++)
- Python 3.8+

### Dependencies

```bash
# macOS (Homebrew)
brew install highs libomp

# Python packages
pip install matplotlib numpy
```

### Optional: LaTeX (for paper compilation)
```bash
brew install --cask mactex  # macOS
```

## Building

```bash
cd src/cpp
make          # Full build (HiGHS + OpenMP)
make no-highs # Without ILP support (heuristics only)
make minimal  # Without HiGHS or OpenMP
```

This produces `src/cpp/cndp_solver`.

### Building IRMS baseline

The IRMS baseline is from Zhou et al. (2023). Their code is available at:
https://github.com/INFORMSJoC/2022.0130

To build:
```bash
# Clone the IRMS repository
git clone https://github.com/INFORMSJoC/2022.0130.git irms_repo

# Build our modified wrapper (adds --direct mode for scripted benchmarks)
cd src/cpp
clang++ -std=c++17 -O3 -march=native -o irms IRMS_modified.cpp
```

## Usage

### Running the solver

```bash
# Usage: cndp_solver <graph_file> <k> <algorithm> [options]
# Algorithms: ilp, degree, betweenness, greedy, ms_ils

# Example: MS-ILS on Bovine graph, remove 12 nodes
./src/cpp/cndp_solver data/icij_colombia/colombia_comp0.txt 20 ms_ils --restarts 25 --time-limit 60

# Example: ILP on small graph
./src/cpp/cndp_solver data/karate.graph 5 ilp --time-limit 120
```

Output is JSON:
```json
{
  "algorithm": "ms_ils",
  "objective": 359,
  "time_seconds": 0.167,
  "k": 20,
  "n": 413,
  "removed_nodes": [45, 102, ...]
}
```

### Running benchmarks

```bash
# Smoke test (~40 seconds, verifies everything works)
python3 src/run_cpp_benchmarks.py --smoke

# Full reproduction of paper results (~2.5 hours)
python3 src/run_final_benchmarks.py
```

### Generating figures

```bash
python3 src/generate_paper_figures.py
# Output in figures/ directory (PDF + PNG)
```

## Reproducing Paper Results

### Step 1: Setup

```bash
git clone https://github.com/danielreales00/cndp-fiscal-evasion.git
cd cndp-fiscal-evasion

# Install dependencies
brew install highs libomp  # macOS
pip install matplotlib numpy

# Build solver
cd src/cpp && make && cd ../..

# Clone and setup IRMS
git clone https://github.com/INFORMSJoC/2022.0130.git irms_repo
clang++ -std=c++17 -O3 -march=native -o src/cpp/irms src/cpp/IRMS_modified.cpp
```

### Step 2: Verify with smoke test

```bash
python3 src/run_cpp_benchmarks.py --smoke
```

Expected output: 3 instances (karate, dolphins, football) solve in ~40s.

### Step 3: Run full benchmarks

```bash
python3 src/run_final_benchmarks.py
```

This runs:
- **12 IRMS paper instances** (Bovine through powergrid, n=121 to n=4941)
- **4 Colombian fiscal evasion instances** (comp0, comp1, full, all)
- Each at k = 5%, 10%, 15% of n
- Total: 48 (instance, k) pairs
- Expected time: ~2.5 hours on Apple M2

Results are saved to `results/benchmark_irms_paper.json` and `results/benchmark_colombia.json`.

### Step 4: Generate figures and compile paper

```bash
python3 src/generate_paper_figures.py
cd . && /Library/TeX/texbin/pdflatex -interaction=nonstopmode paper.tex
/Library/TeX/texbin/pdflatex -interaction=nonstopmode paper.tex  # second pass for refs
```

## Datasets

### Included in this repository

| Dataset | Nodes | Edges | Description |
|---------|-------|-------|-------------|
| colombia_comp0 | 413 | 437 | Largest connected cluster of Colombian offshore entities |
| colombia_comp1 | 339 | 358 | Second largest cluster |
| colombia_full | 3,429 | 3,624 | All clusters with n ≥ 50 |
| colombia_all | 5,847 | 5,922 | Entire Colombian subgraph |
| karate | 34 | 78 | Zachary's karate club (test) |
| dolphins | 62 | 159 | Dolphin social network (test) |
| football | 115 | 613 | NCAA football (test) |

### IRMS benchmark instances (from Zhou et al. 2023)

Download from: https://github.com/INFORMSJoC/2022.0130

After cloning the IRMS repo, instances are at `irms_repo/data/realworld/`:
Bovine (n=121), Circuit (n=252), Treni_Roma (n=255), Ecoli (n=328), USAir97 (n=332), humanDiseasome (n=516), Hamilton1000 (n=1000), EU_flights (n=1191), openflights (n=1858), yeast1 (n=2018), facebook (n=4039), powergrid (n=4941).

### ICIJ Offshore Leaks raw data (optional)

The Colombian graphs were extracted from the ICIJ Offshore Leaks database:
https://offshoreleaks.icij.org/

Raw CSV download (~660MB):
```bash
mkdir -p data/icij && cd data/icij
curl -O https://offshoreleaks-data.icij.org/offshoreleaks/csv/full-oldb.LATEST.zip
unzip full-oldb.LATEST.zip
```

## Key Results

| Instance type | MS-ILS vs IRMS gap | Notes |
|---|---|---|
| Sparse, n ≤ 516 | 0.0–0.9% | MS-ILS matches IRMS |
| Colombian evasion (sparse) | 0.0–2.7% | Near-optimal on target domain |
| Dense, n > 1000 | 2–49% | IRMS's incremental eval dominates |

## Algorithm Details

See `paper.pdf` Section 5 for full CLRS-style specification of all subroutines with complexity analysis.

**MS-ILS total complexity:** O(R · L · I · k · n · (n + m))

Where R = restarts, L = ILS iterations, I = local search iterations, k = budget, n = nodes, m = edges.

## References

1. Arulselvan, A., Commander, C.W., Elefteriadou, L., Pardalos, P.M. (2009). "Detecting critical nodes in sparse graphs." *Computers & Operations Research*, 36(7):2193–2200.

2. Zhou, Y., Li, J., Hao, J.-K., Glover, F. (2023). "Detecting Critical Nodes in Sparse Graphs via 'Reduce-Solve-Combine' Memetic Search." *INFORMS Journal on Computing*, 36(1):39–60. Code: https://github.com/INFORMSJoC/2022.0130

3. ICIJ Offshore Leaks Database. https://offshoreleaks.icij.org/ (Licensed under ODbL)

## License

MIT License. See [LICENSE](LICENSE).

The ICIJ Offshore Leaks data is licensed under the Open Database License (ODbL).
The IRMS implementation (Zhou et al.) has its own license — see their repository.
