# Growing Virtual Materials: Cellular Automata Meets Geostatistics

Hybrid **Plurigaussian Simulation (PGS)** + empirical **Karnaugh-map Cellular
Automata (CA)** for growing synthetic three-phase porous-media microstructures.

## Layout
- `pgsca/` - the package
  - `pgs_tools.py` - Gaussian fields + threshold classification (forward PGS)
  - `karnaugh_tools.py` - K-map engine (build/apply table, sequential
    simulation) and morphology metrics
  - `hybrid_tools.py` - anisotropy inversion + PGS-seeded refinement
- `scripts/validate_realdata.py` - end-to-end check against a real material
- `notebooks/k-map-practice.ipynb` - development + results notebook
- `data/ThreePhase.tif` - real three-phase micro-CT volume
- `archive/rule_based_ca/` - earlier explicit-rule CA track (reference only)
- `reference/gstools-tutorial/` - third-party gstools tutorial

## Install
```
pip install -e .
```
Or, without installing, notebooks self-bootstrap the repo root onto `sys.path`.

## Quick start
```python
from pgsca import make_gaussian_fields, make_lithotype_map, build_table, \
    sequential_simulate, compute_morphology, recover_plurigaussian, make_pgs_seed
```

## Project goals
1. Build a hybrid PGS + CA framework that grows synthetic microstructures. (done)
2. Understand how classification rules control morphology, connectivity,
   complexity. (tooling + metrics in place)
3. Solve the inverse problem: target microstructure -> rules/parameters.
   (PGS parameters recoverable; CA-rule inversion open)
