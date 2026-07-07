"""
hybrid_tools.py

Combines the two completed pieces of the inverse problem into one place:

1. estimate_anisotropy()  - recovers the underlying PGS distribution
   (angle alpha, correlation lengths L_major/L_minor) directly from a
   real binary image, via directional variograms + Gaussian-anamorphosis
   correction. No search, no seed dependence.

2. karnaugh_tools imports - recovers the local neighbourhood-pattern rule
   directly from real data (build_table), and the tools to run/measure it
   (apply_table, sequential_simulate, compute_morphology).

This is the workspace for combining the two: seeding sequential_simulate
with a PGS-recovered field instead of random noise, so the Karnaugh table
only has to do local refinement instead of building spatial structure
from nothing. The combination function itself is not written yet -
this file currently just gathers both complete, working pieces so we can
build on top of them.
"""

import numpy as np
import gstools as gs

from pgs_tools import make_gaussian_fields
from karnaugh_tools import build_table, apply_table, sequential_simulate, compute_morphology


# ---------------------------------------------------------------------
# 1. PGS parameter recovery (variogram-based, arcsine-corrected)
# ---------------------------------------------------------------------

def _corrected_gamma(B, theta, sampling_size, sampling_seed, bin_edges):
    """Gaussian-anamorphosis / arcsine correction for an indicator variogram."""
    grid_size = B.shape[0]
    x = y = np.arange(grid_size)
    bin_center, gamma_I = gs.vario_estimate(
        (x, y), B.astype(float), bin_edges=bin_edges, angles=theta, mesh_type='structured',
        sampling_size=sampling_size, sampling_seed=sampling_seed
    )
    p = B.mean()
    sill_I = p * (1 - p)
    rho_I = np.clip(1 - gamma_I / sill_I, -0.999, 0.999)
    rho_Z = np.sin(rho_I * np.pi / 2)
    return bin_center, 1 - rho_Z


def _fit_range(B, theta, sampling_size, sampling_seed, bin_edges, corrected, weights=None):
    grid_size = B.shape[0]
    if corrected:
        bin_center, gamma = _corrected_gamma(B, theta, sampling_size, sampling_seed, bin_edges)
    else:
        x = y = np.arange(grid_size)
        bin_center, gamma = gs.vario_estimate(
            (x, y), B.astype(float), bin_edges=bin_edges, angles=theta, mesh_type='structured',
            sampling_size=sampling_size, sampling_seed=sampling_seed
        )
    model = gs.Gaussian(dim=2)
    model.fit_variogram(bin_center, gamma, anis=False, weights=weights)
    return model.len_scale


def estimate_anisotropy(
    B, coarse_n_angles=18, fine_span_deg=15, fine_n_angles=21,
    coarse_sampling_size=4000, fine_sampling_size=6000,
    final_sampling_size=10000, sampling_seed=1,
    max_dist_frac=0.7, n_bins=25, search_weights='inv',
    help=False
):
    if help:
        print("""
        estimate_anisotropy() parameters:
        B                    : 2D binary numpy array (0/1) - the field to analyse
        coarse_n_angles      : number of angles in the initial full-sweep search (default 18)
        fine_span_deg        : half-width in degrees of the fine search window (default 15)
        fine_n_angles        : number of angles in the fine search (default 21)
        coarse_sampling_size : point-pairs sampled per angle in coarse sweep (default 4000)
        fine_sampling_size   : point-pairs sampled per angle in fine sweep (default 6000)
        final_sampling_size  : point-pairs sampled for the final major/minor fits (default 10000)
        sampling_seed        : seed for variogram point-pair sampling (default 1)
        max_dist_frac        : max lag distance as a fraction of grid size (default 0.7)
        n_bins               : number of variogram distance bins (default 25)
        search_weights       : weighting passed to fit_variogram during angle search (default 'inv')

        Returns (alpha_star, L_major_star, L_minor_star,
                 coarse_angles, coarse_ranges, fine_angles, fine_ranges)

        Recovers orientation angle and major/minor correlation lengths
        directly from a real binary field B, using a coarse-to-fine
        directional variogram sweep (angle search) followed by a
        Gaussian-anamorphosis-corrected range fit (L_major) and an
        uncorrected perpendicular fit (L_minor).
        """)
        return

    grid_size = B.shape[0]
    bin_edges = np.linspace(0, max_dist_frac * grid_size, n_bins)

    # coarse angle sweep
    coarse_angles = np.linspace(0, np.pi, coarse_n_angles, endpoint=False)
    coarse_ranges = np.array([
        _fit_range(B, th, coarse_sampling_size, sampling_seed, bin_edges, corrected=False, weights=search_weights)
        for th in coarse_angles
    ])
    coarse_best = coarse_angles[np.argmax(coarse_ranges)]

    # fine angle sweep around the coarse best
    half_span = np.deg2rad(fine_span_deg)
    fine_angles = np.linspace(coarse_best - half_span, coarse_best + half_span, fine_n_angles) % np.pi
    fine_ranges = np.array([
        _fit_range(B, th, fine_sampling_size, sampling_seed, bin_edges, corrected=False, weights=search_weights)
        for th in fine_angles
    ])
    alpha_star = fine_angles[np.argmax(fine_ranges)]

    # final major/minor length scale fits
    L_major_star = _fit_range(B, alpha_star, final_sampling_size, sampling_seed, bin_edges, corrected=True)
    perp_theta = (alpha_star + np.pi / 2) % np.pi
    L_minor_star = _fit_range(B, perp_theta, final_sampling_size, sampling_seed, bin_edges, corrected=False)

    return alpha_star, L_major_star, L_minor_star, coarse_angles, coarse_ranges, fine_angles, fine_ranges


# ---------------------------------------------------------------------
# 2. Karnaugh / MPS table recovery - imported directly from karnaugh_tools:
#    build_table, apply_table, sequential_simulate, compute_morphology
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# 3. Combination (PGS-seeded K-map refinement) - not written yet.
#    This is the next piece to build.
# ---------------------------------------------------------------------