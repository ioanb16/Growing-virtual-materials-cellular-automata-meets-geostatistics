"""
descriptors.py
==============
A suite of microstructure descriptors for comparing binary 2D fields.

Each descriptor takes a field (2D array of 0/1) and returns a fingerprint.
`describe()` runs them all and returns one flat labelled dict.
`build_table()` + `pca_suite()` compare a stack of fields.

Descriptors, low-order to rich:
    0. interface_density, two_point_corr  low-order (proportion / roughness / S2)
    1. fourier_spectrum                   what scale is the structure at
    2. wavelet_energy                     energy per scale, split by direction
    3. persistence                        topological fingerprint (blobs & holes)
    4. minkowski                          area, perimeter, Euler (periodic)

Dependencies: numpy, scipy, PyWavelets (pywt), gudhi
    pip install numpy scipy PyWavelets gudhi

Convention: phase of interest is field == 1 (foreground).
All topology measures are computed periodically (wrap-around edges).
"""

import numpy as np
import pywt
import gudhi
from scipy.ndimage import gaussian_filter


# ---------------------------------------------------------------------------
# 0. Low-order descriptors  -  the cheap "are these the same field?" checks
# ---------------------------------------------------------------------------
# proportion (field.mean()), interface density and the S2 covariance curve are
# the low-order measures: fast, and enough to catch gross differences in phase
# fraction and boundary roughness. They MISS connectivity / arrangement, which
# is exactly why the richer descriptors below (persistence, Minkowski) exist -
# two fields can match on proportion + interface + S2 yet look nothing alike.
def interface_density(field):
    """Fraction of periodic neighbour pairs that differ (boundary-length proxy).
    Higher = rougher / more finely mixed; lower = smoother, blobbier."""
    f = np.asarray(field)
    right = (f != np.roll(f, -1, axis=1)).mean()
    down = (f != np.roll(f, -1, axis=0)).mean()
    return 0.5 * (right + down)


def two_point_corr(field, rmax=None):
    """Radially-averaged two-point covariance S2(r) via periodic FFT
    autocorrelation. S2(r) = P(two cells a distance r apart are BOTH phase 1).
    Returns (r, S2). S2(0) = area fraction; it decays to area^2 at large r.
    """
    f = np.asarray(field, dtype=float)
    n = f.size
    F = np.fft.fft2(f)
    ac = np.fft.fftshift(np.fft.ifft2(F * np.conj(F)).real / n)   # S2 at every lag
    size = f.shape[0]
    c = size // 2
    yy, xx = np.mgrid[0:size, 0:size]
    rr = np.hypot(yy - c, xx - c).astype(int)
    if rmax is None:
        rmax = size // 2
    S2 = np.array([ac[rr == r].mean() for r in range(rmax + 1)])
    return np.arange(rmax + 1), S2


# ---------------------------------------------------------------------------
# 1. Fourier power spectrum  -  what scale is the structure at?
# ---------------------------------------------------------------------------
def fourier_spectrum(field, nbins=None):
    """Radially-averaged power spectrum of a 2D field.
    Returns (freq, power). Low freq = large-scale structure, high freq = fine.
    """
    f = np.asarray(field, dtype=float)
    f = f - f.mean()                       # drop DC so it doesn't dominate

    F = np.fft.fftshift(np.fft.fft2(f))
    power2d = np.abs(F) ** 2

    ny, nx = f.shape
    cy, cx = ny // 2, nx // 2
    y, x = np.indices((ny, nx))
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2).astype(int)

    if nbins is None:
        nbins = r.max() + 1
    tbin = np.bincount(r.ravel(), weights=power2d.ravel(), minlength=nbins)
    nr = np.bincount(r.ravel(), minlength=nbins)
    radial = tbin / np.maximum(nr, 1)
    return np.arange(len(radial)), radial


# ---------------------------------------------------------------------------
# 2. Wavelet energy  -  energy per scale, split by direction (H/V/D)
# ---------------------------------------------------------------------------
def wavelet_energy(field, wavelet="db2", levels=4):
    """Split a 2D field into scale layers, measure energy in each.
    Returns {levelN: {horizontal, vertical, diagonal, total}}.
    NOTE: level 1 is the FINEST scale, level `levels` the COARSEST.
    """
    f = np.asarray(field, dtype=float)
    f = f - f.mean()

    coeffs = pywt.wavedec2(f, wavelet=wavelet, level=levels)
    out = {}
    for i, detail in enumerate(coeffs[1:], start=1):
        cH, cV, cD = detail
        eH, eV, eD = np.sum(cH ** 2), np.sum(cV ** 2), np.sum(cD ** 2)
        out[f"level{i}"] = {"horizontal": eH, "vertical": eV,
                            "diagonal": eD, "total": eH + eV + eD}
    return out


# ---------------------------------------------------------------------------
# 3. Persistent homology  -  topological fingerprint (blobs & holes)
# ---------------------------------------------------------------------------
def persistence(field, smooth=1.0):
    """Persistent homology via a cubical complex (lower-star filtration).
    Floods the image; records birth/death of blobs (H0) and holes (H1).
    Returns {blobs_H0/holes_H1: {n_features, pairs, lifetimes, max_lifetime}}.
    """
    f = np.asarray(field, dtype=float)
    f = gaussian_filter(f, smooth)

    cc = gudhi.CubicalComplex(top_dimensional_cells=f)
    cc.persistence()

    out = {}
    for dim, name in [(0, "blobs_H0"), (1, "holes_H1")]:
        pairs = cc.persistence_intervals_in_dimension(dim)
        pairs = pairs[np.isfinite(pairs[:, 1])]        # drop the never-dying feature
        lifetimes = pairs[:, 1] - pairs[:, 0]
        out[name] = {"n_features": len(pairs), "pairs": pairs,
                     "lifetimes": lifetimes,
                     "max_lifetime": lifetimes.max() if len(lifetimes) else 0.0}
    return out


# ---------------------------------------------------------------------------
# 4. Minkowski functionals  -  area, perimeter, Euler (periodic)
# ---------------------------------------------------------------------------
def minkowski(field):
    """The three 2D Minkowski functionals of the phase (field == 1), periodic.
      area      = fraction of cells that are 1
      perimeter = boundary length per cell
      euler     = blobs - holes (8-connected, 2x2 bit-quad estimator)
    Returns a dict {area, perimeter, euler}.
    """
    f = (np.asarray(field) > 0).astype(int)
    ny, nx = f.shape

    area = f.mean()

    right = f != np.roll(f, -1, axis=1)
    down = f != np.roll(f, -1, axis=0)
    perimeter = (right.sum() + down.sum()) / (ny * nx)

    a = f
    b = np.roll(f, -1, axis=1)
    c = np.roll(f, -1, axis=0)
    d = np.roll(np.roll(f, -1, axis=0), -1, axis=1)
    s = a + b + c + d
    q1 = np.sum(s == 1)
    q3 = np.sum(s == 3)
    qd = np.sum((s == 2) & (a == d) & (b == c) & (a != b))
    euler = (q1 - q3 + 2 * qd) / 4.0

    return {"area": area, "perimeter": perimeter, "euler": euler}


# ---------------------------------------------------------------------------
# Summaries: squeeze each descriptor to a few numbers (for the harness / PCA)
# ---------------------------------------------------------------------------
def _lowobj_summary(field):
    """Low-order fingerprint: phase proportion, interface density, and a
    correlation length read off the S2 curve (lag where S2 has decayed halfway
    from its peak S2(0) to its large-lag floor ~area^2)."""
    area = float(np.asarray(field).mean())
    iface = interface_density(field)
    r, s2 = two_point_corr(field)
    floor = area ** 2
    half = floor + 0.5 * (s2[0] - floor)
    below = np.where(s2 <= half)[0]
    corr_len = float(below[0]) if len(below) else float(r[-1])
    return {"low_proportion": area, "low_interface": iface,
            "low_corr_len": corr_len}


def _fourier_summary(field):
    freq, power = fourier_spectrum(field)
    freq, power = freq[1:], power[1:]                  # drop DC
    char_freq = np.sum(freq * power) / np.sum(power)   # "blob size ruler"
    mask = power > 0
    slope = np.polyfit(np.log(freq[mask]), np.log(power[mask]), 1)[0]
    return {"fourier_char_freq": char_freq, "fourier_slope": slope}


def _wavelet_summary(field):
    e = wavelet_energy(field)
    out = {}
    for lvl, vals in e.items():
        out[f"wave_{lvl}_total"] = vals["total"]
        hvd = np.array([vals["horizontal"], vals["vertical"], vals["diagonal"]], float)
        out[f"wave_{lvl}_imbalance"] = hvd.std() / (hvd.mean() + 1e-9)
    return out


def _persistence_summary(field):
    p = persistence(field)
    out = {}
    for name in ["blobs_H0", "holes_H1"]:
        out[f"pers_{name}_n"] = p[name]["n_features"]
        out[f"pers_{name}_maxlife"] = p[name]["max_lifetime"]
    return out


def _minkowski_summary(field):
    m = minkowski(field)
    return {"mink_area": m["area"], "mink_perimeter": m["perimeter"],
            "mink_euler": m["euler"]}


# ---------------------------------------------------------------------------
# The harness: run everything, return one flat labelled fingerprint
# ---------------------------------------------------------------------------
def describe(field, verbose=True):
    """Run the whole descriptor suite on one field.
    Returns a single flat labelled dict of summary numbers (the fingerprint).
    """
    out = {}
    out.update(_lowobj_summary(field))
    out.update(_fourier_summary(field))
    out.update(_wavelet_summary(field))
    out.update(_persistence_summary(field))
    out.update(_minkowski_summary(field))

    if verbose:
        print("descriptor suite:")
        for k, v in out.items():
            print(f"  {k:28s} {v:.4f}")
    return out


# ---------------------------------------------------------------------------
# PCA over a stack of fingerprints  -  which axes carry the variation?
# ---------------------------------------------------------------------------
def build_table(fields):
    """Run describe() on each field, stack into a table.
    Returns (matrix [n_images x n_descriptors], column labels).
    """
    rows, labels = [], None
    for f in fields:
        d = describe(f, verbose=False)
        if labels is None:
            labels = list(d.keys())
        rows.append([d[k] for k in labels])
    return np.array(rows, float), labels


def pca_suite(matrix, labels, n_components=2, verbose=True):
    """Standardise columns, run PCA (via SVD), report variance explained and
    the descriptors loading most strongly on each component.
    Returns (scores [n_images x n_components], variance_ratio).
    """
    X = matrix.copy()
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Xs = (X - mu) / sd

    U, S, Vt = np.linalg.svd(Xs, full_matrices=False)
    var = S ** 2 / (len(Xs) - 1)
    var_ratio = var / var.sum()
    scores = U * S

    if verbose:
        print("variance explained:")
        for i in range(n_components):
            print(f"  PC{i+1}: {var_ratio[i]*100:5.1f}%")
            loading = Vt[i]
            for j in np.argsort(-np.abs(loading))[:4]:
                print(f"      {labels[j]:28s} {loading[j]:+.3f}")

    return scores[:, :n_components], var_ratio