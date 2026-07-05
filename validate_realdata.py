"""Validate the full recovery + combination pipeline against ThreePhase.tif.

Closes Definition-of-Done item 3 (one real material). Run from the repo root:
    python validate_realdata.py
Expected (seed-fixed): combined-Moore phase 0 near prop 0.375, 120 components,
mean area 205, no percolation, vs real slice 0.370 / 137 / 177 / no perc.
"""
import numpy as np, tifffile
from karnaugh_tools import build_table, sequential_simulate, compute_morphology
from hybrid_tools import recover_plurigaussian, make_pgs_seed

lut = np.zeros(256, int); lut[0], lut[128], lut[255] = 0, 1, 2
vol = lut[tifffile.imread('ThreePhase.tif')]        # phases 0/128/255 -> 0/1/2

target = vol[128]
train  = [vol[k] for k in range(0, 256, 26)][:10]   # 10 spread-out slices

rec  = recover_plurigaussian(target)
print('recovered:', rec)
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
