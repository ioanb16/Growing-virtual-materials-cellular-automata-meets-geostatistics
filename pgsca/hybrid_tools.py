"""
hybrid_tools.py -- as per Appendix C of the handoff document.
"""
import numpy as np
import gstools as gs

from .pgs_tools import make_gaussian_fields
from .karnaugh_tools import build_table, apply_table, sequential_simulate, compute_morphology


def _corrected_gamma(B, theta, sampling_size, sampling_seed, bin_edges):
    """Gaussian-anamorphosis / arcsine correction for an indicator variogram."""
    grid_size = B.shape[0]
    x = y = np.arange(grid_size)
    bin_center, gamma_I = gs.vario_estimate(
        (x, y), B.astype(float), bin_edges=bin_edges, angles=theta,
        mesh_type='structured',
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
            (x, y), B.astype(float), bin_edges=bin_edges, angles=theta,
            mesh_type='structured',
            sampling_size=sampling_size, sampling_seed=sampling_seed
        )
    model = gs.Gaussian(dim=2)
    model.fit_variogram(bin_center, gamma, anis=False, weights=weights)
    return model.len_scale




def _auto_max_dist_frac(gamma_source, grid_size, cap=0.7, floor=0.05):
    """Pick a lag window ~5x the omnidirectional correlation range.

    gamma_source : callable(bin_edges) -> (bin_center, gamma)
    On short-range textures, binning lags out to 0.7*grid puts nearly every
    bin at the sill, so the directional range fits (and hence the angle
    sweep) are dominated by sill noise. Restricting the window to a few
    correlation lengths concentrates the bins where the variogram actually
    carries directional information.
    """
    import gstools as _gs
    probe_edges = np.linspace(0, cap * grid_size, 25)
    bin_center, gamma = gamma_source(probe_edges)
    model = _gs.Gaussian(dim=2)
    model.fit_variogram(bin_center, gamma, anis=False)
    L0 = model.len_scale
    return float(np.clip(5.0 * L0 / grid_size, floor, cap))

def estimate_anisotropy(
    B, coarse_n_angles=18, fine_span_deg=15, fine_n_angles=21,
    coarse_sampling_size=4000, fine_sampling_size=6000,
    final_sampling_size=10000, sampling_seed=1,
    max_dist_frac='auto', n_bins=25, search_weights='inv',
    help=False
):
    if help:
        print("""
        estimate_anisotropy(): recovers (alpha, L_major, L_minor) for one
        binary field via directional variograms + arcsine correction.
        See handoff Appendix C.
        """)
        return
    grid_size = B.shape[0]
    if max_dist_frac == 'auto':
        x = y = np.arange(grid_size)
        def _omni(edges):
            return gs.vario_estimate((x, y), B.astype(float), bin_edges=edges,
                                     mesh_type='structured',
                                     sampling_size=coarse_sampling_size,
                                     sampling_seed=sampling_seed)
        max_dist_frac = _auto_max_dist_frac(_omni, grid_size)
    bin_edges = np.linspace(0, max_dist_frac * grid_size, n_bins)

    # coarse angle sweep
    coarse_angles = np.linspace(0, np.pi, coarse_n_angles, endpoint=False)
    coarse_ranges = np.array([
        _fit_range(B, th, coarse_sampling_size, sampling_seed, bin_edges,
                   corrected=False, weights=search_weights)
        for th in coarse_angles
    ])
    coarse_best = coarse_angles[np.argmax(coarse_ranges)]

    # fine angle sweep around the coarse best
    half_span = np.deg2rad(fine_span_deg)
    fine_angles = np.linspace(coarse_best - half_span, coarse_best + half_span,
                              fine_n_angles) % np.pi
    fine_ranges = np.array([
        _fit_range(B, th, fine_sampling_size, sampling_seed, bin_edges,
                   corrected=False, weights=search_weights)
        for th in fine_angles
    ])
    alpha_star = fine_angles[np.argmax(fine_ranges)]

    # final major/minor length scale fits
    L_major_star = _fit_range(B, alpha_star, final_sampling_size, sampling_seed,
                              bin_edges, corrected=True)
    perp_theta = (alpha_star + np.pi / 2) % np.pi
    L_minor_star = _fit_range(B, perp_theta, final_sampling_size, sampling_seed,
                              bin_edges, corrected=True)

    return (alpha_star, L_major_star, L_minor_star,
            coarse_angles, coarse_ranges, fine_angles, fine_ranges)


# ---------------------------------------------------------------------
# 3. Combination (PGS-seeded K-map refinement) -- Task 1
# ---------------------------------------------------------------------

def make_pgs_seed(shape, params_1, params_2, proportions,
                  seed_1=0, seed_2=1, help=False):
    if help:
        print("""
        make_pgs_seed() parameters:
        shape       : (rows, cols) of the seed grid (must be square, rows==cols)
        params_1    : (alpha, L_major, L_minor) for Gaussian field 1
        params_2    : (alpha, L_major, L_minor) for Gaussian field 2
        proportions : [p0, p1, p2] observed phase proportions; converted to
                      threshold cuts inside make_lithotype_map
        seed_1/2    : random seeds for the two fields

        Regenerates a three-phase lithotype map from recovered PGS
        parameters, for use as the initial_grid of sequential_simulate.
        Returns a 2D integer array of states 0/1/2.
        """)
        return
    from .pgs_tools import make_lithotype_map
    rows, cols = shape
    assert rows == cols, "pgs_tools generates square grids"
    a1, Lmaj1, Lmin1 = params_1
    a2, Lmaj2, Lmin2 = params_2
    f1, f2 = make_gaussian_fields(
        grid_size=rows,
        len_scale_1=[Lmaj1, Lmin1], angles_1=[a1, a1], seed_1=seed_1,
        len_scale_2=[Lmaj2, Lmin2], angles_2=[a2, a2], seed_2=seed_2,
    )
    p0, p1, p2 = proportions
    return make_lithotype_map(f1, f2, Mat1=p0, Mat2=p1, Mat3=p2).astype(int)


# ---------------------------------------------------------------------
# 4. Multi-field plurigaussian recovery -- Task 3
# ---------------------------------------------------------------------

def _fit_range_masked(values, pos, p, theta, sampling_size, sampling_seed,
                      bin_edges, corrected, weights=None,
                      angles_tol=np.pi/16, bandwidth=8.0):
    """Directional Gaussian-range fit on an *unstructured* (masked) indicator.

    values : 1D array of indicator values (0/1) at the masked pixels
    pos    : (rows_idx, cols_idx) float coordinates of the masked pixels
    p      : indicator proportion among the masked pixels (for the sill
             p(1-p) used by the arcsine correction)
    """
    bin_center, gamma_I = gs.vario_estimate(
        pos, values, bin_edges=bin_edges, angles=theta,
        angles_tol=angles_tol, bandwidth=bandwidth,
        sampling_size=sampling_size, sampling_seed=sampling_seed,
        mesh_type='unstructured'
    )
    if corrected:
        sill_I = p * (1 - p)
        rho_I = np.clip(1 - gamma_I / sill_I, -0.999, 0.999)
        gamma = 1 - np.sin(rho_I * np.pi / 2)
    else:
        gamma = gamma_I
    model = gs.Gaussian(dim=2)
    model.fit_variogram(bin_center, gamma, anis=False, weights=weights)
    return model.len_scale


def estimate_anisotropy_masked(
    I, mask, coarse_n_angles=18, fine_span_deg=15, fine_n_angles=21,
    coarse_sampling_size=2500, fine_sampling_size=3000,
    final_sampling_size=5000, sampling_seed=1,
    max_dist_frac='auto', n_bins=25, search_weights='inv',
    help=False
):
    if help:
        print("""
        estimate_anisotropy_masked() parameters:
        I    : 2D binary numpy array (0/1) -- the indicator to analyse
        mask : 2D boolean array -- only pixels where mask is True are used.
               Pairs are formed only between masked pixels, so the variogram
               is the *conditional* indicator variogram on the masked region.
        (remaining parameters as in estimate_anisotropy)

        Same coarse-to-fine directional-variogram recovery as
        estimate_anisotropy, but computed on an unstructured point set
        (the masked pixels) instead of the full structured grid. Used to
        recover the SECOND Gaussian field of a plurigaussian map, whose
        indicator (phase 2 vs phase 1) is only defined where field 1 has
        already been classified away from phase 0.

        Returns (alpha_star, L_major_star, L_minor_star).
        """)
        return
    grid_size = I.shape[0]
    ii, jj = np.nonzero(mask)
    pos = (ii.astype(float), jj.astype(float))
    values = I[mask].astype(float)
    p = values.mean()
    if max_dist_frac == 'auto':
        def _omni(edges):
            return gs.vario_estimate(pos, values, bin_edges=edges,
                                     mesh_type='unstructured',
                                     sampling_size=coarse_sampling_size,
                                     sampling_seed=sampling_seed)
        max_dist_frac = _auto_max_dist_frac(_omni, grid_size, cap=0.5)
    bin_edges = np.linspace(0, max_dist_frac * grid_size, n_bins)

    coarse_angles = np.linspace(0, np.pi, coarse_n_angles, endpoint=False)
    coarse_ranges = np.array([
        _fit_range_masked(values, pos, p, th, coarse_sampling_size,
                          sampling_seed, bin_edges, corrected=False,
                          weights=search_weights)
        for th in coarse_angles
    ])
    coarse_best = coarse_angles[np.argmax(coarse_ranges)]

    half_span = np.deg2rad(fine_span_deg)
    fine_angles = np.linspace(coarse_best - half_span, coarse_best + half_span,
                              fine_n_angles) % np.pi
    fine_ranges = np.array([
        _fit_range_masked(values, pos, p, th, fine_sampling_size,
                          sampling_seed, bin_edges, corrected=False,
                          weights=search_weights)
        for th in fine_angles
    ])
    alpha_star = fine_angles[np.argmax(fine_ranges)]

    L_major_star = _fit_range_masked(values, pos, p, alpha_star,
                                     final_sampling_size, sampling_seed,
                                     bin_edges, corrected=True)
    perp_theta = (alpha_star + np.pi / 2) % np.pi
    L_minor_star = _fit_range_masked(values, pos, p, perp_theta,
                                     final_sampling_size, sampling_seed,
                                     bin_edges, corrected=True)
    return alpha_star, L_major_star, L_minor_star


def recover_plurigaussian(real_map, sampling_seed=1, help=False, **kwargs):
    if help:
        print("""
        recover_plurigaussian() parameters:
        real_map : 2D integer array of phases 0/1/2, produced by (or assumed
                   to follow) make_lithotype_map's hierarchical logic:
                       phase 0 :  field_1 <  cut_1
                       phase 1 :  field_1 >= cut_1  and  field_2 <  cut_2
                       phase 2 :  field_1 >= cut_1  and  field_2 >= cut_2
        kwargs   : forwarded to the two anisotropy estimators.

        Recovers the FULL plurigaussian parameter set:
          proportions  -> the two threshold cuts (cut_1 = ppf(p0),
                          cut_2 = ppf(p1/(p1+p2)), exactly inverting
                          make_lithotype_map)
          field 1      -> (alpha_1, L_major_1, L_minor_1) from the phase-0
                          indicator on the full grid (phase 0 is a function
                          of field 1 ALONE under the hierarchy above)
          field 2      -> (alpha_2, L_major_2, L_minor_2) from the phase-2
                          indicator restricted to the mask (phase != 0),
                          where phase 2 vs phase 1 is a function of field 2
                          ALONE, and independence of the two fields makes the
                          masked variogram an unbiased estimate of field 2's
                          indicator variogram.

        Returns dict with keys: proportions, cut_1, cut_2, params_1, params_2.
        """)
        return
    from scipy.stats import norm
    real_map = real_map.astype(int)
    counts = np.bincount(real_map.ravel(), minlength=3).astype(float)
    props = counts / counts.sum()
    cut_1 = norm.ppf(props[0])
    cut_2 = norm.ppf(props[1] / (props[1] + props[2]))

    # field 1: phase-0 indicator, full grid (structured)
    B0 = (real_map == 0).astype(int)
    a1, L1maj, L1min = estimate_anisotropy(
        B0, sampling_seed=sampling_seed, **kwargs)[:3]

    # field 2: phase-2 indicator conditional on phase != 0 (unstructured)
    mask = real_map != 0
    I2 = (real_map == 2).astype(int)
    a2, L2maj, L2min = estimate_anisotropy_masked(
        I2, mask, sampling_seed=sampling_seed)

    return dict(proportions=props.tolist(), cut_1=float(cut_1),
                cut_2=float(cut_2),
                params_1=(float(a1), float(L1maj), float(L1min)),
                params_2=(float(a2), float(L2maj), float(L2min)))