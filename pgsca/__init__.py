"""pgsca - Plurigaussian Simulation + Karnaugh-map Cellular Automata.

Hybrid framework for growing synthetic three-phase porous-media
microstructures. See the top-level README for the pipeline overview.
"""
from .pgs_tools import (make_gaussian_fields, make_lithotype_map,
                        plot_fields, plot_lithotype_map)
from .karnaugh_tools import (encode_neighbourhood, build_table, apply_table,
                             sequential_simulate, compute_morphology)
from .hybrid_tools import (estimate_anisotropy, estimate_anisotropy_masked,
                           recover_plurigaussian, recover_plurigaussian_multi,
                           make_pgs_seed)

__all__ = [
    "make_gaussian_fields", "make_lithotype_map", "plot_fields",
    "plot_lithotype_map", "encode_neighbourhood", "build_table", "apply_table",
    "sequential_simulate", "compute_morphology", "estimate_anisotropy",
    "estimate_anisotropy_masked", "recover_plurigaussian",
    "recover_plurigaussian_multi", "make_pgs_seed",
]
__version__ = "0.1.0"
