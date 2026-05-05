"""
Final benchmark suite for CNDP paper.
Instances from IRMS paper (Zhou et al. 2023) + ICIJ Colombian fiscal evasion network.

ILP only runs on n <= 62 (beyond that: O(n^3) constraints make it infeasible).
MS-ILS budget: up to 10 minutes per instance.
IRMS: state-of-the-art baseline.

Usage:
    python3 run_final_benchmarks.py
"""
import subprocess
import json
import re
import time
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
IRMS_REPO = BASE_DIR / "irms_repo" / "data" / "realworld"
IRMS_SYNTH = BASE_DIR / "irms_repo" / "data" / "synthetic"
ICIJ_DIR = DATA_DIR / "icij_colombia"
RESULTS_DIR = BASE_DIR / "results"
CPP_DIR = Path(__file__).parent / "cpp"

CNDP_SOLVER = CPP_DIR / "cndp_solver"
IRMS_BINARY = CPP_DIR / "irms"

RESULTS_DIR.mkdir(exist_ok=True)

# ILP is feasible only for very small instances
ILP_MAX_N = 62
# MS-ILS time budget per instance (seconds)
MSILS_BUDGET = 600

# Instances from Zhou et al. 2023 paper (these have known best results)
# k values follow the paper's convention: k = ceil(n * ratio), ratio in {0.05, 0.10, 0.15}
IRMS_INSTANCES = [
    {"name": "Bovine",          "file": IRMS_REPO / "Bovine.txt",         "n": 121},
    {"name": "Circuit",         "file": IRMS_REPO / "Circuit.txt",        "n": 252},
    {"name": "Ecoli",           "file": IRMS_REPO / "Ecoli.txt",          "n": 328},
    {"name": "USAir97",         "file": IRMS_REPO / "USAir97.txt",        "n": 332},
    {"name": "Treni_Roma",      "file": IRMS_REPO / "Treni_Roma.txt",     "n": 255},
    {"name": "humanDiseasome",  "file": IRMS_REPO / "humanDiseasome.txt", "n": 516},
    {"name": "Hamilton1000",    "file": IRMS_REPO / "Hamilton1000.txt",   "n": 1000},
    {"name": "EU_flights",      "file": IRMS_REPO / "EU_flights.txt",     "n": 1191},
    {"name": "openflights",     "file": IRMS_REPO / "openflights.txt",    "n": 1858},
    {"name": "yeast1",          "file": IRMS_REPO / "yeast1.txt",         "n": 2018},
    {"name": "facebook",        "file": IRMS_REPO / "facebook.txt",       "n": 4039},
    {"name": "powergrid",       "file": IRMS_REPO / "powergrid.txt",      "n": 4941},
]

# Colombian fiscal evasion instances
COLOMBIA_INSTANCES = [
    {"name": "colombia_comp0",  "file": ICIJ_DIR / "colombia_comp0.txt",  "irms_file": ICIJ_DIR / "colombia_comp0_irms.txt",  "n": 413},
    {"name": "colombia_comp1",  "file": ICIJ_DIR / "colombia_comp1.txt",  "irms_file": ICIJ_DIR / "colombia_comp1_irms.txt",  "n": 339},
    {"name": "colombia_full",   "file": ICIJ_DIR / "colombia_full.txt",   "irms_file": ICIJ_DIR / "colombia_full_irms.txt",   "n": 3429},
    {"name": "colombia_all",    "file": ICIJ_DIR / "colombia_all.txt",    "irms_file": ICIJ_DIR / "colombia_all_irms.txt",    "n": 5847},
]


def get_k_values(n):
    """k = 5%, 10%, 15% of n (following IRMS paper convention)."""
    return [max(3, int(n * r)) for r in [0.05, 0.10, 0.15]]


def get_restarts(n, budget_seconds):
    """Estimate restarts that fit within budget based on observed scaling."""
    if n <= 200:
        return 30
    elif n <= 500:
        return 20
    elif n <= 1000:
        return 10
    elif n <= 3000:
        return 5
    else:
        return 3


def run_cndp_solver(instance_file, k, algorithm, time_limit=300, restarts=25, threads=8):
    cmd = [
        str(CNDP_SOLVER), str(instance_file), str(k), algorithm,
        "--time-limit", str(time_limit),
        "--restarts", str(restarts),
        "--threads", str(threads),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=time_limit + 30)
        if result.returncode != 0:
            print(f"      [WARN] cndp_solver failed: {result.stderr[:200]}")
            return None
        return json.loads(result.stdout.strip())
    except subprocess.TimeoutExpired:
        print(f"      [TIMEOUT] {algorithm} exceeded {time_limit}s")
        return {"algorithm": algorithm, "objective": -1, "time_seconds": time_limit, "timeout": True}
    except Exception as e:
        print(f"      [ERROR] {e}")
        return None


def run_irms(irms_file, k, time_limit=30, repeats=3):
    cmd = [str(IRMS_BINARY), "--direct", str(irms_file), str(k), str(time_limit), str(repeats)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=time_limit * repeats + 60)
        if result.returncode != 0:
            print(f"      [WARN] irms failed: {result.stderr[:200]}")
            return None
        lines = result.stdout.strip().split('\n')
        best_obj = None
        best_time = None
        for line in lines:
            if "BestObjValue" in line:
                obj_match = re.search(r'BestObjValue\s*=\s*(\d+)', line)
                time_match = re.search(r'BestTime\s*=\s*([\d.eE+-]+)', line)
                if obj_match:
                    obj = int(obj_match.group(1))
                    t = float(time_match.group(1)) if time_match else 0.0
                    if best_obj is None or obj < best_obj:
                        best_obj = obj
                        best_time = t
            elif "best value" in line:
                obj_match = re.search(r'best value\s*=\s*(\d+)', line)
                time_match = re.search(r'average time\s*=\s*([\d.eE+-]+)', line)
                if obj_match:
                    obj = int(obj_match.group(1))
                    t = float(time_match.group(1)) if time_match else best_time
                    if best_obj is None or obj < best_obj:
                        best_obj = obj
                        best_time = t
        if best_obj is not None:
            return {"algorithm": "irms", "objective": best_obj, "time_seconds": best_time}
        print(f"      [WARN] Could not parse IRMS output")
        return None
    except subprocess.TimeoutExpired:
        print(f"      [TIMEOUT] irms exceeded budget")
        return None
    except Exception as e:
        print(f"      [ERROR] irms: {e}")
        return None


def print_header(msg):
    print(f"\n{'='*80}")
    print(f"  {msg}")
    print(f"{'='*80}")


def run_instance(name, file_path, irms_file, n, k):
    """Run all applicable algorithms on one (instance, k) pair."""
    row = {"instance": name, "n": n, "k": k}
    restarts = get_restarts(n, MSILS_BUDGET)

    # ILP: only for small instances
    if n <= ILP_MAX_N:
        ilp = run_cndp_solver(file_path, k, "ilp", time_limit=180)
        if ilp:
            row["ilp"] = {"obj": ilp["objective"], "time": ilp["time_seconds"]}
            print(f"      ILP:         obj={ilp['objective']:>10}  time={ilp['time_seconds']:.3f}s")

    # Heuristics: degree, betweenness, greedy
    for alg in ["degree", "betweenness", "greedy"]:
        r = run_cndp_solver(file_path, k, alg, time_limit=60)
        if r:
            row[alg] = {"obj": r["objective"], "time": r["time_seconds"]}

    # MS-ILS: our algorithm
    msils_tl = min(MSILS_BUDGET, max(60, n // 2))
    msils = run_cndp_solver(file_path, k, "ms_ils", restarts=restarts, time_limit=msils_tl)
    if msils:
        row["ms_ils"] = {"obj": msils["objective"], "time": msils["time_seconds"]}
        print(f"      MS-ILS:      obj={msils['objective']:>10}  time={msils['time_seconds']:.3f}s")

    # IRMS: state-of-the-art
    if irms_file and irms_file.exists():
        irms_tl = 30 if n <= 2000 else 60
        irms = run_irms(irms_file, k, time_limit=irms_tl, repeats=3)
        if irms:
            row["irms"] = {"obj": irms["objective"], "time": irms["time_seconds"]}
            print(f"      IRMS:        obj={irms['objective']:>10}  time={irms['time_seconds']:.4f}s")

    # Summary line
    msils_obj = row.get("ms_ils", {}).get("obj", "?")
    irms_obj = row.get("irms", {}).get("obj", "?")
    if isinstance(msils_obj, int) and isinstance(irms_obj, int) and irms_obj > 0:
        gap = (msils_obj - irms_obj) / irms_obj * 100
        print(f"      Gap MS-ILS vs IRMS: {gap:+.2f}%")

    return row


def benchmark_irms_paper():
    """Run on instances from Zhou et al. 2023."""
    print_header("BENCHMARK: IRMS Paper Instances (Zhou et al. 2023)")
    print(f"  ILP cutoff: n <= {ILP_MAX_N}")
    print(f"  MS-ILS budget: {MSILS_BUDGET}s max")
    results = []

    for inst in IRMS_INSTANCES:
        if not inst["file"].exists():
            print(f"  [SKIP] {inst['name']} not found")
            continue

        k_values = get_k_values(inst["n"])
        for k in k_values:
            print(f"\n  {inst['name']} (n={inst['n']}, k={k}):", flush=True)
            row = run_instance(
                inst["name"], inst["file"],
                inst["file"],  # IRMS binary reads the same format
                inst["n"], k
            )
            results.append(row)

    with open(RESULTS_DIR / "benchmark_irms_paper.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: results/benchmark_irms_paper.json")
    return results


def benchmark_colombia():
    """Run on ICIJ Colombian fiscal evasion networks."""
    print_header("BENCHMARK: Colombian Fiscal Evasion Network (ICIJ)")
    results = []

    for inst in COLOMBIA_INSTANCES:
        if not inst["file"].exists():
            print(f"  [SKIP] {inst['name']} not found")
            continue

        k_values = get_k_values(inst["n"])
        for k in k_values:
            print(f"\n  {inst['name']} (n={inst['n']}, k={k}):", flush=True)
            row = run_instance(
                inst["name"], inst["file"],
                inst.get("irms_file"),
                inst["n"], k
            )
            results.append(row)

    with open(RESULTS_DIR / "benchmark_colombia.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: results/benchmark_colombia.json")
    return results


if __name__ == "__main__":
    total_start = time.time()

    print("\n" + "=" * 80)
    print("  CNDP FINAL BENCHMARK SUITE")
    print("  Hardware: Apple M2 | 8 cores | 16GB RAM")
    print(f"  Solver: {CNDP_SOLVER}")
    print(f"  IRMS:   {IRMS_BINARY}")
    print(f"  ILP cutoff: n <= {ILP_MAX_N} (O(n^3) constraints infeasible beyond this)")
    print(f"  MS-ILS budget: up to {MSILS_BUDGET}s per instance")
    print("=" * 80)

    if not CNDP_SOLVER.exists():
        print(f"\n[ERROR] cndp_solver not found. Run: cd src/cpp && make")
        sys.exit(1)
    if not IRMS_BINARY.exists():
        print(f"\n[ERROR] irms not found. Compile IRMS_modified.cpp first")
        sys.exit(1)

    all_results = {}
    all_results["irms_paper"] = benchmark_irms_paper()
    all_results["colombia"] = benchmark_colombia()

    total_time = time.time() - total_start

    with open(RESULTS_DIR / "final_benchmarks.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print_header(f"ALL BENCHMARKS COMPLETE — Total: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Results in: {RESULTS_DIR}/")
