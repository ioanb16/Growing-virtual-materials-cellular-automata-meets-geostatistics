"""Validate the full recovery + combination pipeline against ThreePhase.tif.

Recovers PGS parameters by averaging over multiple real slices (up to a time
budget), builds a Moore K-map from spread-out slices, seeds with the recovered
parameters, refines, and prints a morphology comparison (proportion / components
/ mean area / largest / percolation) for real vs. PGS-seed vs. combined, per
phase. Run from the repo root:
    python scripts/validate_realdata.py
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np, tifffile
from pgsca.karnaugh_tools import build_table, sequential_simulate, compute_morphology
from pgsca.hybrid_tools import recover_plurigaussian_multi, make_pgs_seed

lut = np.zeros(256, int); lut[0], lut[128], lut[255] = 0, 1, 2
vol = lut[tifffile.imread(ROOT/'data'/'ThreePhase.tif')]        # phases 0/128/255 -> 0/1/2

target = vol[128]
train  = [vol[k] for k in range(0, 256, 26)][:10]   # 10 spread-out slices

# Recover PGS parameters from as many slices as fit within the time budget,
# then average them.  Increase TIME_BUDGET (seconds) for more slices / higher
# accuracy; set to None to process every slice in `pgs_slices`.
TIME_BUDGET = 120          # seconds
pgs_slices  = [vol[k] for k in range(0, 256, 13)]   # up to ~20 spread-out slices

rec = recover_plurigaussian_multi(pgs_slices, time_budget=TIME_BUDGET)
print(f"recovered (averaged over {rec['n_slices_used']} slice(s)):", rec)
seed = make_pgs_seed(target.shape, rec['params_1'], rec['params_2'],
                     rec['proportions'], seed_1=11, seed_2=12)
tm   = build_table([(s, s) for s in train], neighbourhood='moore')
g, _ = sequential_simulate(tm, target.shape, proportions=rec['proportions'],
                           rng=np.random.default_rng(1), initial_grid=seed)

for name, grid in [('real', target), ('pgs seed', seed), ('combined', g)]:
    for ph in range(3):
        m = compute_morphology(grid)[ph]
        print(f"{name:9s} phase {ph}: prop={m['proportion']:.3f} "
              f"n={m['n_components']:4d} mean={m['mean_area']:7.1f} "
              f"largest={m['largest_area']:6d} perc={m['percolates']}")
