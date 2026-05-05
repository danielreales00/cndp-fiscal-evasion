"""Generate publication-quality figures for the CNDP paper."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

RESULTS = Path(__file__).parent.parent / "results"
FIGURES = Path(__file__).parent.parent / "figures"
FIGURES.mkdir(exist_ok=True)

with open(RESULTS / "benchmark_irms_paper.json") as f:
    irms_data = json.load(f)
with open(RESULTS / "benchmark_colombia.json") as f:
    col_data = json.load(f)

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# ── Fig 5: Quality gap bar chart (k ≈ 10%n) ──────────────────────────────
def fig_quality_gap():
    # Pick k ≈ 10%n entries
    k10 = {}
    for r in irms_data + col_data:
        ratio = r['k'] / r['n']
        if 0.08 <= ratio <= 0.12:
            name = r['instance']
            ms = r.get('ms_ils', {}).get('obj')
            ir = r.get('irms', {}).get('obj')
            if ms and ir and ir > 0 and ms > 0:
                gap = (ms - ir) / ir * 100
                k10[name] = gap

    names = list(k10.keys())
    gaps = list(k10.values())
    colors = ['#2ecc71' if g <= 1 else '#f39c12' if g <= 10 else '#e74c3c' for g in gaps]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars = ax.bar(range(len(names)), gaps, color=colors, edgecolor='white', linewidth=0.5)
    ax.axhline(y=0, color='black', linewidth=0.8, linestyle='-')
    ax.axhline(y=1, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.axhline(y=10, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Gap to IRMS (%)')
    ax.set_title('MS-ILS vs IRMS: Solution Quality Gap at $k = 0.10n$')

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='Gap ≤ 1%'),
        Patch(facecolor='#f39c12', label='1% < Gap ≤ 10%'),
        Patch(facecolor='#e74c3c', label='Gap > 10%'),
    ]
    ax.legend(handles=legend_elements, loc='upper left')
    ax.set_ylim(bottom=min(-2, min(gaps) - 1))

    fig.savefig(FIGURES / "fig5_quality_gap.pdf")
    fig.savefig(FIGURES / "fig5_quality_gap.png")
    plt.close(fig)
    print("  fig5_quality_gap done")


# ── Fig 6: Scalability (time vs n) ────────────────────────────────────────
def fig_scalability():
    ms_n, ms_t, ir_n, ir_t = [], [], [], []
    labels = {}

    for r in irms_data + col_data:
        ratio = r['k'] / r['n']
        if not (0.08 <= ratio <= 0.12):
            continue
        n = r['n']
        ms = r.get('ms_ils', {})
        ir = r.get('irms', {})
        if ms.get('time') and ms['time'] > 0 and ms.get('obj', -1) > 0:
            ms_n.append(n)
            ms_t.append(ms['time'])
            labels[n] = r['instance']
        if ir.get('time') and ir['time'] > 0:
            ir_n.append(n)
            ir_t.append(ir['time'])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(ms_n, ms_t, s=60, c='#3498db', marker='o', label='MS-ILS', zorder=3)
    ax.scatter(ir_n, ir_t, s=60, c='#e74c3c', marker='^', label='IRMS', zorder=3)

    for n, t in zip(ms_n, ms_t):
        if n in labels and (n > 1500 or n < 200 or n in [332, 516]):
            ax.annotate(labels[n], (n, t), textcoords="offset points",
                       xytext=(5, 5), fontsize=7, alpha=0.7)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Number of nodes ($n$)')
    ax.set_ylabel('Wall-clock time (seconds)')
    ax.set_title('Scalability: MS-ILS vs IRMS at $k = 0.10n$')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(FIGURES / "fig6_scalability.pdf")
    fig.savefig(FIGURES / "fig6_scalability.png")
    plt.close(fig)
    print("  fig6_scalability done")


# ── Fig 7: Colombia results ───────────────────────────────────────────────
def fig_colombia():
    instances = ['colombia_comp0', 'colombia_comp1', 'colombia_full', 'colombia_all']
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharey=False)

    for idx, inst_name in enumerate(instances):
        inst_data = [r for r in col_data if r['instance'] == inst_name]
        k_vals = [r['k'] for r in inst_data]
        ms_objs = [r.get('ms_ils', {}).get('obj', 0) for r in inst_data]
        ir_objs = [r.get('irms', {}).get('obj', 0) for r in inst_data]

        x = np.arange(len(k_vals))
        w = 0.35
        axes[idx].bar(x - w/2, ms_objs, w, label='MS-ILS', color='#3498db')
        axes[idx].bar(x + w/2, ir_objs, w, label='IRMS', color='#e74c3c')
        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels([str(k) for k in k_vals], fontsize=9)
        axes[idx].set_xlabel('$k$')
        n = inst_data[0]['n'] if inst_data else '?'
        axes[idx].set_title(f'{inst_name}\n($n={n}$)', fontsize=10)
        if idx == 0:
            axes[idx].set_ylabel('Objective (connected pairs)')
            axes[idx].legend(fontsize=8)

    fig.suptitle('Colombian Fiscal Evasion Networks: MS-ILS vs IRMS', fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig7_colombia.pdf")
    fig.savefig(FIGURES / "fig7_colombia.png")
    plt.close(fig)
    print("  fig7_colombia done")


# ── Fig 8: Density vs Gap ─────────────────────────────────────────────────
def fig_density_gap():
    densities, gaps, names = [], [], []

    # We need m/n for each instance. Compute from the data.
    # Approximate m from benchmark data — we stored n but not m directly.
    # Use known values:
    known_m = {
        'Bovine': 190, 'Circuit': 399, 'Treni_Roma': 312, 'Ecoli': 456,
        'USAir97': 2126, 'humanDiseasome': 1188, 'Hamilton1000': 1000,
        'EU_flights': 6782, 'openflights': 9604, 'yeast1': 2930,
        'facebook': 88234, 'powergrid': 6594,
        'colombia_comp0': 437, 'colombia_comp1': 358,
        'colombia_full': 3624, 'colombia_all': 5922,
    }

    for r in irms_data + col_data:
        ms = r.get('ms_ils', {}).get('obj')
        ir = r.get('irms', {}).get('obj')
        if ms and ir and ir > 0 and ms > 0:
            gap = (ms - ir) / ir * 100
            m = known_m.get(r['instance'], r['n'])
            density = m / r['n']
            densities.append(density)
            gaps.append(gap)
            names.append(f"{r['instance']}(k={r['k']})")

    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(densities, gaps, s=40, c='#3498db', alpha=0.7, edgecolors='white', linewidth=0.5)

    # Annotate outliers (gap > 20%)
    for d, g, n in zip(densities, gaps, names):
        if g > 20:
            ax.annotate(n, (d, g), textcoords="offset points",
                       xytext=(5, 3), fontsize=7, alpha=0.7)

    # Trend line
    z = np.polyfit(densities, gaps, 1)
    x_line = np.linspace(min(densities), max(densities), 100)
    ax.plot(x_line, np.polyval(z, x_line), '--', color='#e74c3c', alpha=0.5, label=f'Linear trend')

    ax.set_xlabel('Graph density ($m/n$)')
    ax.set_ylabel('Gap MS-ILS vs IRMS (%)')
    ax.set_title('Effect of Graph Density on Algorithm Performance Gap')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=-5)

    fig.savefig(FIGURES / "fig8_density_gap.pdf")
    fig.savefig(FIGURES / "fig8_density_gap.png")
    plt.close(fig)
    print("  fig8_density_gap done")


# ── Fig 9: All algorithms comparison ──────────────────────────────────────
def fig_all_algorithms():
    target = ['Bovine', 'Ecoli', 'USAir97', 'humanDiseasome', 'EU_flights', 'powergrid']
    algs = ['degree', 'betweenness', 'greedy', 'ms_ils', 'irms']
    alg_labels = ['Degree', 'Betweenness', 'Greedy', 'MS-ILS (ours)', 'IRMS (SOTA)']
    colors = ['#95a5a6', '#bdc3c7', '#7f8c8d', '#3498db', '#e74c3c']

    rows = []
    for inst in target:
        for r in irms_data:
            ratio = r['k'] / r['n']
            if r['instance'] == inst and 0.08 <= ratio <= 0.12:
                rows.append(r)
                break

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(rows))
    width = 0.15

    for i, (alg, label, color) in enumerate(zip(algs, alg_labels, colors)):
        vals = []
        for r in rows:
            v = r.get(alg, {}).get('obj', 0)
            vals.append(v if v > 0 else 0.1)
        offset = (i - 2) * width
        ax.bar(x + offset, vals, width, label=label, color=color, edgecolor='white', linewidth=0.3)

    ax.set_xticks(x)
    xlabels = [f"{r['instance']}\n($n={r['n']}, k={r['k']}$)" for r in rows]
    ax.set_xticklabels(xlabels, fontsize=9)
    ax.set_ylabel('Objective value (connected pairs)')
    ax.set_yscale('log')
    ax.set_title('Algorithm Comparison at $k = 0.10n$')
    ax.legend(fontsize=9, ncol=3, loc='upper right')
    ax.grid(True, alpha=0.2, axis='y')

    fig.tight_layout()
    fig.savefig(FIGURES / "fig9_algorithms.pdf")
    fig.savefig(FIGURES / "fig9_algorithms.png")
    plt.close(fig)
    print("  fig9_algorithms done")


if __name__ == "__main__":
    print("Generating paper figures...")
    fig_quality_gap()
    fig_scalability()
    fig_colombia()
    fig_density_gap()
    fig_all_algorithms()
    print(f"Done. Figures in {FIGURES}/")
