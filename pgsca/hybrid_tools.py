"""
hybrid_tools.py -- Plurigaussian parameter recovery and Anisotropy estimation.
"""
import numpy as np
import gstools as gs
from scipy.stats import norm
from .pgs_tools import make_gaussian_fields, make_lithotype_map

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
    """Fits a 2D Gaussian variogram model to find the correlation length scale."""
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


def _fit_range_rose(B, angles, sampling_size, sampling_seed, bin_edges, corrected, weights=None):
    """Directional Gaussian range at each angle -- a 'rose' of ranges."""
    return np.array([
        _fit_range(B, th, sampling_size, sampling_seed, bin_edges, corrected=corrected, weights=weights)
        for th in angles
    ])


def _ellipse_orientation(angles, ranges):
    """
    Fits a centred ellipse to a rose of directional variogram ranges using linear least-squares.
    Returns (alpha, L_major, L_minor); alpha in [0, pi).
    """
    th = np.asarray(angles, float)
    R  = np.asarray(ranges, float)
    good = R > 1e-9
    th, R = th[good], R[good]
    
    u = 1.0 / R**2
    M = np.column_stack([np.cos(th)**2, np.sin(th)**2, 2*np.sin(th)*np.cos(th)])
    a, b, h = np.linalg.lstsq(M, u, rcond=None)[0]
    
    Q = np.array([[a, h], [h, b]])
    evals, evecs = np.linalg.eigh(Q)
    evals = np.clip(evals, 1e-12, None)
    lengths = 1.0 / np.sqrt(evals)
    
    i_major = int(np.argmax(lengths))
    L_major, L_minor = lengths[i_major], lengths[1 - i_major]
    vx, vy = evecs[:, i_major]
    alpha = np.arctan2(vy, vx) % np.pi
    
    return alpha, L_major, L_minor


def _auto_max_dist_frac(gamma_source, grid_size, cap=0.7, floor=0.05):
    """Picks a lag window ~5x the omnidirectional correlation range to avoid sill noise."""
    probe_edges = np.linspace(0, cap * grid_size, 25)
    bin_center, gamma = gamma_source(probe_edges)
    model = gs.Gaussian(dim=2)
    model.fit_variogram(bin_center, gamma, anis=False)
    return float(np.clip(5.0 * model.len_scale / grid_size, floor, cap))


def estimate_anisotropy(
    B, n_angles=24, sampling_size=6000, final_sampling_size=10000,
    sampling_seed=1, max_dist_frac='auto', n_bins=25, search_weights='inv'
):
    """
    Recovers (alpha, L_major, L_minor) for one binary field.
    Returns (alpha_star, L_major_star, L_minor_star, sweep_angles, sweep_ranges).
    """
    grid_size = B.shape[0]
    if max_dist_frac == 'auto':
        x = y = np.arange(grid_size)
        def _omni(edges):
            return gs.vario_estimate((x, y), B.astype(float), bin_edges=edges,
                                     mesh_type='structured',
                                     sampling_size=sampling_size,
                                     sampling_seed=sampling_seed)
        max_dist_frac = _auto_max_dist_frac(_omni, grid_size)
    bin_edges = np.linspace(0, max_dist_frac * grid_size, n_bins)

    sweep_angles = np.linspace(0, np.pi, n_angles, endpoint=False)
    sweep_ranges = _fit_range_rose(B, sweep_angles, sampling_size, sampling_seed,
                                   bin_edges, corrected=False, weights=search_weights)
    alpha_star, _, _ = _ellipse_orientation(sweep_angles, sweep_ranges)

    L_major_star = _fit_range(B, alpha_star, final_sampling_size, sampling_seed,
                              bin_edges, corrected=True)
    perp = (alpha_star + np.pi / 2) % np.pi
    L_minor_star = _fit_range(B, perp, final_sampling_size, sampling_seed,
                              bin_edges, corrected=True)

    return alpha_star, L_major_star, L_minor_star, sweep_angles, sweep_ranges


def make_pgs_seed(shape, params_1, params_2, proportions, seed_1=0, seed_2=1):
    """Regenerates a three-phase lithotype map from recovered PGS parameters."""
    rows, cols = shape
    assert rows == cols, "pgs_tools generates square grids"
    a1, Lmaj1, Lmin1 = params_1
    a2, Lmaj2, Lmin2 = params_2
    
    f1, f2 = make_gaussian_fields(
        grid_size=rows,
        len_scale_1=[Lmaj1, Lmin1], angles_1=[a1, a1], seed_1=seed_1,
        len_scale_2=[Lmaj2, Lmin2], angles_2=[a2, a2], seed_2=seed_2,
    )
    return make_lithotype_map(f1, f2, Mat1=proportions[0], Mat2=proportions[1], Mat3=proportions[2]).astype(int)


def _fit_range_masked(values, pos, p, theta, sampling_size, sampling_seed,
                      bin_edges, corrected, weights=None,
                      angles_tol=np.pi/16, bandwidth=8.0):
    """Directional Gaussian-range fit on an unstructured (masked) indicator."""
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
    I, mask, n_angles=24, sampling_size=3000, final_sampling_size=5000,
    sampling_seed=1, max_dist_frac='auto', n_bins=25, search_weights='inv'
):
    """Field-2 recovery on the masked (unstructured) point set."""
    grid_size = I.shape[0]
    ii, jj = np.nonzero(mask)
    pos = (ii.astype(float), jj.astype(float))
    values = I[mask].astype(float)
    p = values.mean()
    
    if max_dist_frac == 'auto':
        def _omni(edges):
            return gs.vario_estimate(pos, values, bin_edges=edges,
                                     mesh_type='unstructured',
                                     sampling_size=sampling_size,
                                     sampling_seed=sampling_seed)
        max_dist_frac = _auto_max_dist_frac(_omni, grid_size, cap=0.5)
    bin_edges = np.linspace(0, max_dist_frac * grid_size, n_bins)

    sweep_angles = np.linspace(0, np.pi, n_angles, endpoint=False)
    sweep_ranges = np.array([
        _fit_range_masked(values, pos, p, th, sampling_size, sampling_seed,
                          bin_edges, corrected=False, weights=search_weights)
        for th in sweep_angles
    ])
    alpha_star, _, _ = _ellipse_orientation(sweep_angles, sweep_ranges)

    L_major_star = _fit_range_masked(values, pos, p, alpha_star, final_sampling_size, sampling_seed, bin_edges, corrected=True)
    perp_theta = (alpha_star + np.pi / 2) % np.pi
    L_minor_star = _fit_range_masked(values, pos, p, perp_theta, final_sampling_size, sampling_seed, bin_edges, corrected=True)
    
    return alpha_star, L_major_star, L_minor_star


def _circular_mean_angle(angles):
    """Circular mean for angles in [0, π) using the double-angle trick."""
    doubled = 2.0 * np.asarray(angles, float)
    mean_doubled = np.arctan2(np.mean(np.sin(doubled)), np.mean(np.cos(doubled)))
    return float(mean_doubled / 2 % np.pi)


def recover_plurigaussian(real_map, sampling_seed=1, **kwargs):
    """
    Recovers the FULL plurigaussian parameter set by reversing the hierarchical truncation.
    Returns dict with keys: proportions, cut_1, cut_2, params_1, params_2.
    """
    real_map = real_map.astype(int)
    counts = np.bincount(real_map.ravel(), minlength=3).astype(float)
    props = counts / counts.sum()
    
    cut_1 = norm.ppf(props[0])
    cut_2 = norm.ppf(props[1] / (props[1] + props[2]))

    B0 = (real_map == 0).astype(int)
    a1, L1maj, L1min = estimate_anisotropy(B0, sampling_seed=sampling_seed, **kwargs)[:3]

    mask = real_map != 0
    I2 = (real_map == 2).astype(int)
    a2, L2maj, L2min = estimate_anisotropy_masked(I2, mask, sampling_seed=sampling_seed, **kwargs)

    return dict(proportions=props.tolist(), cut_1=float(cut_1), cut_2=float(cut_2),
                params_1=(float(a1), float(L1maj), float(L1min)),
                params_2=(float(a2), float(L2maj), float(L2min)))


def recover_plurigaussian_multi(slices, time_budget=None, sampling_seed=1, **kwargs):
    """
    Runs recover_plurigaussian on each slice in `slices`, stopping early when
    `time_budget` seconds have elapsed (if given), then returns the averaged
    parameters.  Angles are averaged with a circular mean so wrapping at π is
    handled correctly.

    Parameters
    ----------
    slices      : iterable of 2-D int arrays — lithotype slices to process.
    time_budget : float | None — wall-clock seconds; stop after this budget is
                  exhausted.  None means process every slice.
    sampling_seed, **kwargs : forwarded to recover_plurigaussian.

    Returns
    -------
    dict with the same keys as recover_plurigaussian plus 'n_slices_used'.
    """
    import time
    results = []
    t0 = time.perf_counter()
    for s in slices:
        results.append(recover_plurigaussian(s, sampling_seed=sampling_seed, **kwargs))
        if time_budget is not None and (time.perf_counter() - t0) >= time_budget:
            break

    if not results:
        raise ValueError("No slices were processed — check that `slices` is non-empty.")

    props  = np.mean([r['proportions'] for r in results], axis=0).tolist()
    cut_1  = float(np.mean([r['cut_1']  for r in results]))
    cut_2  = float(np.mean([r['cut_2']  for r in results]))
    a1     = _circular_mean_angle([r['params_1'][0] for r in results])
    Lmaj1  = float(np.mean([r['params_1'][1] for r in results]))
    Lmin1  = float(np.mean([r['params_1'][2] for r in results]))
    a2     = _circular_mean_angle([r['params_2'][0] for r in results])
    Lmaj2  = float(np.mean([r['params_2'][1] for r in results]))
    Lmin2  = float(np.mean([r['params_2'][2] for r in results]))

    return dict(
        proportions=props, cut_1=cut_1, cut_2=cut_2,
        params_1=(a1, Lmaj1, Lmin1),
        params_2=(a2, Lmaj2, Lmin2),
        n_slices_used=len(results),
    )