"""
Unified C++ benchmark runner for CNDP paper.
Calls cndp_solver (ILP, degree, betweenness, greedy, ms_ils) and irms across all instances.
Collects JSON results for plotting.

Usage:
    python3 run_cpp_benchmarks.py [--quick]   # quick mode skips large ILP instances

Expected time: ~3-8 minutes on Apple M2 (full), ~1 min (quick)
"""
import subprocess
import json
import time
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
IRMS_DIR = DATA_DIR / "irms_format"
RESULTS_DIR = BASE_DIR / "results"
CPP_DIR = Path(__file__).parent / "cpp"

CNDP_SOLVER = CPP_DIR / "cndp_solver"
IRMS_BINARY = CPP_DIR / "irms"

RESULTS_DIR.mkdir(exist_ok=True)

INSTANCES = [
    {"name": "karate",   "file": DATA_DIR / "karate.graph",   "irms": IRMS_DIR / "karate.txt",   "n": 34,  "k_values": [3, 5, 8]},
    {"name": "dolphins", "file": DATA_DIR / "dolphins.graph", "irms": IRMS_DIR / "dolphins.txt", "n": 62,  "k_values": [5, 8, 12]},
    {"name": "football", "file": DATA_DIR / "football.graph", "irms": IRMS_DIR / "football.txt", "n": 115, "k_values": [8, 12, 20]},
    {"name": "jazz",     "file": DATA_DIR / "jazz.graph",     "irms": IRMS_DIR / "jazz.txt",     "n": 198, "k_values": [10, 20, 30]},
    {"name": "USAir",    "file": DATA_DIR / "USAir.txt",      "irms": IRMS_DIR / "USAir.txt",    "n": 332, "k_values": [15, 30, 50]},
    {"name": "power",    "file": DATA_DIR / "power.graph",    "irms": IRMS_DIR / "power.txt",    "n": 4941,"k_values": [50, 100, 200]},
]


def get_restarts_for_n(n):
    if n <= 100:
        return 30
    elif n <= 300:
        return 20
    elif n <= 1000:
        return 10
    else:
        return 3

ILP_MAX_N = 70


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
            print(f"    [WARN] cndp_solver failed: {result.stderr[:200]}")
            return None
        output = result.stdout.strip()
        return json.loads(output)
    except subprocess.TimeoutExpired:
        print(f"    [TIMEOUT] {algorithm} on {instance_file} k={k}")
        return None
    except Exception as e:
        print(f"    [ERROR] {e}")
        return None


def run_irms(irms_file, k, time_limit=10, repeats=5):
    cmd = [
        str(IRMS_BINARY), "--direct", str(irms_file),
        str(k), str(time_limit), str(repeats),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=time_limit * repeats + 60)
        if result.returncode != 0:
            print(f"    [WARN] irms failed: {result.stderr[:200]}")
            return None
        lines = result.stdout.strip().split('\n')
        best_obj = None
        best_time = None
        # Parse IRMS output format:
        # "Repeat = 1, BestObjValue = 45, BestTime = 0.000058, BestGens = 0"
        # "best value = 45, average value = 45, average time = 4.65e-05, succ times = 2"
        for line in lines:
            if "BestObjValue" in line:
                import re
                obj_match = re.search(r'BestObjValue\s*=\s*(\d+)', line)
                time_match = re.search(r'BestTime\s*=\s*([\d.eE+-]+)', line)
                if obj_match:
                    obj = int(obj_match.group(1))
                    t = float(time_match.group(1)) if time_match else 0.0
                    if best_obj is None or obj < best_obj:
                        best_obj = obj
                        best_time = t
            elif "best value" in line:
                import re
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
        print(f"    [WARN] Could not parse IRMS output: {lines[:5]}")
        return None
    except subprocess.TimeoutExpired:
        print(f"    [TIMEOUT] irms on {irms_file} k={k}")
        return None
    except Exception as e:
        print(f"    [ERROR] irms: {e}")
        return None


def print_header(msg):
    print(f"\n{'='*70}")
    print(f"  {msg}")
    print(f"{'='*70}")


def benchmark_quality(quick=False):
    """Compare all algorithms on instances where ILP is feasible."""
    print_header("BENCHMARK 1: Solution Quality (ILP vs MS-ILS vs IRMS)")
    results = []

    for inst in INSTANCES:
        if inst["n"] > ILP_MAX_N:
            continue
        for k in inst["k_values"]:
            print(f"  {inst['name']} (n={inst['n']}, k={k})...", flush=True)
            row = {"instance": inst["name"], "n": inst["n"], "k": k}
            restarts = get_restarts_for_n(inst["n"])

            ilp = run_cndp_solver(inst["file"], k, "ilp", time_limit=180)
            if ilp:
                row["ilp"] = {"obj": ilp["objective"], "time": ilp["time_seconds"]}
                print(f"    ILP: obj={ilp['objective']} time={ilp['time_seconds']:.3f}s")

            msils = run_cndp_solver(inst["file"], k, "ms_ils", restarts=restarts)
            if msils:
                row["ms_ils"] = {"obj": msils["objective"], "time": msils["time_seconds"]}
                print(f"    MS-ILS: obj={msils['objective']} time={msils['time_seconds']:.3f}s")

            deg = run_cndp_solver(inst["file"], k, "degree")
            if deg:
                row["degree"] = {"obj": deg["objective"], "time": deg["time_seconds"]}

            bet = run_cndp_solver(inst["file"], k, "betweenness")
            if bet:
                row["betweenness"] = {"obj": bet["objective"], "time": bet["time_seconds"]}

            grdy = run_cndp_solver(inst["file"], k, "greedy")
            if grdy:
                row["greedy"] = {"obj": grdy["objective"], "time": grdy["time_seconds"]}

            if inst["irms"].exists():
                irms = run_irms(inst["irms"], k, time_limit=10, repeats=5)
                if irms:
                    row["irms"] = {"obj": irms["objective"], "time": irms["time_seconds"]}
                    print(f"    IRMS: obj={irms['objective']} time={irms['time_seconds']:.4f}s")

            results.append(row)

    with open(RESULTS_DIR / "cpp_quality_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: results/cpp_quality_results.json")
    return results


def benchmark_scalability(quick=False):
    """Compare heuristics on all instances (no ILP for large ones)."""
    print_header("BENCHMARK 2: Scalability across all instances")
    results = []

    algorithms = ["degree", "betweenness", "greedy", "ms_ils"]

    for inst in INSTANCES:
        k = inst["k_values"][1]  # middle k value
        print(f"  {inst['name']} (n={inst['n']}, k={k})...", flush=True)
        row = {"instance": inst["name"], "n": inst["n"], "k": k}

        for alg in algorithms:
            restarts = get_restarts_for_n(inst["n"])
            time_limit = 300 if alg == "ilp" else 120
            r = run_cndp_solver(inst["file"], k, alg, time_limit=time_limit, restarts=restarts)
            if r:
                row[alg] = {"obj": r["objective"], "time": r["time_seconds"]}
                print(f"    {alg:>12s}: obj={r['objective']:>8} time={r['time_seconds']:.3f}s")

        if inst["n"] <= ILP_MAX_N:
            ilp = run_cndp_solver(inst["file"], k, "ilp", time_limit=180)
            if ilp:
                row["ilp"] = {"obj": ilp["objective"], "time": ilp["time_seconds"]}
                print(f"    {'ilp':>12s}: obj={ilp['objective']:>8} time={ilp['time_seconds']:.3f}s")

        if inst["irms"].exists():
            irms_tl = 30 if inst["n"] > 1000 else 10
            irms = run_irms(inst["irms"], k, time_limit=irms_tl, repeats=3)
            if irms:
                row["irms"] = {"obj": irms["objective"], "time": irms["time_seconds"]}
                print(f"    {'irms':>12s}: obj={irms['objective']:>8} time={irms['time_seconds']:.4f}s")

        results.append(row)

    with open(RESULTS_DIR / "cpp_scalability_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: results/cpp_scalability_results.json")
    return results


def benchmark_k_sensitivity(quick=False):
    """Vary k on a medium instance (football n=115)."""
    print_header("BENCHMARK 3: Budget Sensitivity (varying k on football)")

    inst = next(i for i in INSTANCES if i["name"] == "football")
    k_values = [4, 8, 12, 16, 20, 25, 30, 35, 40]
    results = []

    for k in k_values:
        print(f"  k={k:>3}...", end=" ", flush=True)
        row = {"instance": inst["name"], "n": inst["n"], "k": k}

        for alg in ["degree", "betweenness", "greedy", "ms_ils"]:
            r = run_cndp_solver(inst["file"], k, alg, restarts=20)
            if r:
                row[alg] = {"obj": r["objective"], "time": r["time_seconds"]}

        if inst["irms"].exists():
            irms = run_irms(inst["irms"], k, time_limit=10, repeats=5)
            if irms:
                row["irms"] = {"obj": irms["objective"], "time": irms["time_seconds"]}

        msils_obj = row.get("ms_ils", {}).get("obj", "?")
        irms_obj = row.get("irms", {}).get("obj", "?")
        print(f"MS-ILS={msils_obj} IRMS={irms_obj}")
        results.append(row)

    with open(RESULTS_DIR / "cpp_k_sensitivity_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: results/cpp_k_sensitivity_results.json")
    return results


def benchmark_speed_comparison(quick=False):
    """Time comparison: C++ MS-ILS vs IRMS on same instances."""
    print_header("BENCHMARK 4: Speed Comparison (C++ MS-ILS vs IRMS)")
    results = []

    for inst in INSTANCES:
        if quick and inst["n"] > 500:
            continue
        k = inst["k_values"][1]
        restarts = get_restarts_for_n(inst["n"])
        print(f"  {inst['name']} (n={inst['n']}, k={k})...", flush=True)
        row = {"instance": inst["name"], "n": inst["n"], "k": k}

        msils = run_cndp_solver(inst["file"], k, "ms_ils", restarts=restarts, time_limit=120)
        if msils:
            row["ms_ils"] = {"obj": msils["objective"], "time": msils["time_seconds"]}

        if inst["irms"].exists():
            irms_tl = 30 if inst["n"] <= 1000 else 60
            irms = run_irms(inst["irms"], k, time_limit=irms_tl, repeats=3)
            if irms:
                row["irms"] = {"obj": irms["objective"], "time": irms["time_seconds"]}

        if row.get("ms_ils") and row.get("irms"):
            speedup = row["ms_ils"]["time"] / max(row["irms"]["time"], 0.0001)
            quality_gap = ((row["ms_ils"]["obj"] - row["irms"]["obj"]) /
                          max(row["irms"]["obj"], 1) * 100)
            print(f"    MS-ILS: obj={row['ms_ils']['obj']} ({row['ms_ils']['time']:.3f}s) | "
                  f"IRMS: obj={row['irms']['obj']} ({row['irms']['time']:.4f}s) | "
                  f"IRMS {speedup:.1f}x faster | gap={quality_gap:+.1f}%")
        results.append(row)

    with open(RESULTS_DIR / "cpp_speed_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: results/cpp_speed_results.json")
    return results


def smoke_test():
    """Fast sanity check: karate + dolphins, all algorithms, ~10 seconds."""
    print_header("SMOKE TEST: Quick validation on small instances")
    results = []

    small = [inst for inst in INSTANCES if inst["n"] <= 115]
    for inst in small:
        k = inst["k_values"][1]
        print(f"  {inst['name']} (n={inst['n']}, k={k})...", flush=True)
        row = {"instance": inst["name"], "n": inst["n"], "k": k}

        for alg in ["degree", "betweenness", "greedy", "ms_ils"]:
            r = run_cndp_solver(inst["file"], k, alg, restarts=5, time_limit=30)
            if r:
                row[alg] = {"obj": r["objective"], "time": r["time_seconds"]}
                print(f"    {alg:>12s}: obj={r['objective']:>6} time={r['time_seconds']:.4f}s")

        if inst["n"] <= ILP_MAX_N:
            r = run_cndp_solver(inst["file"], k, "ilp", time_limit=10)
            if r:
                row["ilp"] = {"obj": r["objective"], "time": r["time_seconds"]}
                print(f"    {'ilp':>12s}: obj={r['objective']:>6} time={r['time_seconds']:.4f}s")

        if inst["irms"].exists():
            irms = run_irms(inst["irms"], k, time_limit=5, repeats=2)
            if irms:
                row["irms"] = {"obj": irms["objective"], "time": irms["time_seconds"]}
                print(f"    {'irms':>12s}: obj={irms['objective']:>6} time={irms['time_seconds']:.6f}s")

        results.append(row)

    with open(RESULTS_DIR / "cpp_smoke_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    return results


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    smoke = "--smoke" in sys.argv
    total_start = time.time()

    print("\n" + "=" * 70)
    print("  CNDP C++ BENCHMARK SUITE")
    print("  Hardware: Apple M2 | 8 cores | 16GB RAM")
    print(f"  Solver: {CNDP_SOLVER}")
    print(f"  IRMS:   {IRMS_BINARY}")
    mode = "SMOKE" if smoke else ("QUICK" if quick else "FULL")
    print(f"  Mode:   {mode}")
    print("=" * 70)

    if not CNDP_SOLVER.exists():
        print(f"\n[ERROR] cndp_solver not found at {CNDP_SOLVER}")
        print("  Run: cd src/cpp && make")
        sys.exit(1)
    if not IRMS_BINARY.exists():
        print(f"\n[ERROR] irms not found at {IRMS_BINARY}")
        print("  Compile IRMS_modified.cpp first")
        sys.exit(1)

    if smoke:
        smoke_test()
    else:
        all_results = {}
        all_results["quality"] = benchmark_quality(quick)
        all_results["scalability"] = benchmark_scalability(quick)
        all_results["k_sensitivity"] = benchmark_k_sensitivity(quick)
        all_results["speed"] = benchmark_speed_comparison(quick)

        with open(RESULTS_DIR / "cpp_all_benchmarks.json", 'w') as f:
            json.dump(all_results, f, indent=2, default=str)

    total_time = time.time() - total_start
    print_header(f"DONE — Total: {total_time:.1f}s")
    print(f"  Results in: {RESULTS_DIR}/")
