"""
generation.py
=============
Making and growing two-phase fields, the counterpart to the repair rule in
`pattern_ca.py`. Two jobs live here:

  1. make_field       - synthesise a smooth two-phase "truth" image from scratch
                        (random noise -> blur to a blob size -> threshold).
  2. grow             - the multiscale (coarse-to-fine) generator: learn the
                        pattern vocabulary from a truth image, then grow a NEW
                        field of the same character starting from pure noise.

The growth strand exists because running the fine 3x3 rule straight from pure
noise collapses to tiny specks (a small window can only "see" small structure).
The fix is coarse-to-fine: shrink the truth, grow big blobs cheaply on the small
grid, blow them back up, de-block the staircase edges, then refine with the fine
rule. See `grow()` below.

Building blocks (all periodic / wrap-around, to match pattern_ca.py):
    make_field        smooth Gaussian field thresholded to a target fraction
    block_downsample  shrink k x by k x k block majority
    upsample          blow up k x (nearest-neighbour) then crop
    smooth_binary     periodic blur + re-threshold (de-blocks, pins proportion)
    grow              the full coarse-to-fine pipeline, one call

Dependencies: numpy, pattern_ca (for the CA rule used inside grow())
"""

import numpy as np
import pattern_ca as P


# ---------------------------------------------------------------------------
# 1. Synthesise a truth image from scratch
# ---------------------------------------------------------------------------
def make_field(size, scale, frac=0.5, seed=0):
    """Smooth periodic Gaussian field thresholded to a target phase fraction.

    Random white noise is blurred with a Gaussian of width `scale` (bigger
    scale -> bigger blobs) then thresholded so exactly `frac` of the cells end
    up white (1). frac=0.5 -> even split. Returns an int 0/1 array (size, size).
    """
    rng = np.random.default_rng(seed)
    white = rng.normal(size=(size, size))
    fx = np.fft.fftfreq(size)[:, None]
    fy = np.fft.fftfreq(size)[None, :]
    freq2 = fx ** 2 + fy ** 2
    filt = np.exp(-((2 * np.pi) ** 2) * freq2 * (scale ** 2) / 2.0)
    field = np.fft.ifft2(np.fft.fft2(white) * filt).real
    thresh = np.quantile(field, 1 - frac)        # keep the top `frac` as white
    return (field > thresh).astype(int)


# ---------------------------------------------------------------------------
# 2. Multiscale helpers (coarse-to-fine)
# ---------------------------------------------------------------------------
def block_downsample(f, k):
    """Shrink by k via k x k block majority. Works for any k: the field is
    periodically padded up to a multiple of k first, so no edge is lost.
    Returns a (ceil(n/k), ceil(n/k)) int 0/1 array."""
    n = f.shape[0]
    sc = -(-n // k)                      # ceil(n/k)
    pad = sc * k - n
    fp = np.pad(f, ((0, pad), (0, pad)), mode="wrap")
    return (fp.reshape(sc, k, sc, k).mean(axis=(1, 3)) > 0.5).astype(int)


def upsample(f, k, out):
    """Blow up by k (nearest-neighbour block replication), then crop back to an
    `out` x `out` field. The inverse direction of block_downsample."""
    up = np.repeat(np.repeat(f, k, axis=0), k, axis=1)
    return up[:out, :out]


def smooth_binary(f, scale, frac):
    """Periodic blur + threshold at TARGET proportion `frac`.
    Run after upsample to kill the dead-straight staircase edges block
    replication leaves behind, and to pin the white fraction back to `frac`
    (fixes any proportion drift). Returns an int 0/1 array the same shape as f.
    """
    n = f.shape[0]
    fx = np.fft.fftfreq(n)[:, None]
    fy = np.fft.fftfreq(n)[None, :]
    filt = np.exp(-((2 * np.pi) ** 2) * (fx ** 2 + fy ** 2) * (scale ** 2) / 2.0)
    blurred = np.fft.ifft2(np.fft.fft2(f.astype(float)) * filt).real
    return (blurred > np.quantile(blurred, 1 - frac)).astype(int)


# ---------------------------------------------------------------------------
# 3. The full coarse-to-fine generator
# ---------------------------------------------------------------------------
def grow(truth, k=2, sweeps=100, seed=0, smooth_scale=1.5, radius=1, verbose=False):
    """Grow a NEW field with the same character as `truth`, from pure noise.

    Coarse-to-fine pipeline (the cell-72 recipe, the best generation result):
      1. shrink `truth` by k and learn its coarse vocabulary
      2. grow big blobs on the small grid starting from pure 50/50 noise
      3. upsample those blobs back to full size
      4. smooth_binary to de-block the edges and pin the proportion
      5. refine the edges with the fine (full-resolution) CA rule
      6. clean_specks

    k is the coarsening factor (grown blobs come out ~k x bigger; try 2-4).
    Returns the grown int 0/1 field, same shape as `truth`.

    NOTE: on symmetric structure heavy noise can converge to the phase-INVERTED
    image, which is still a valid reconstruction - compare structure (S2,
    Minkowski, ...) not raw pixel difference.
    """
    n = truth.shape[0]
    frac = float(truth.mean())

    # --- coarse stage: learn + grow on a shrunken copy ---
    coarse_truth = block_downsample(truth, k)
    seen_c = P.build_windows_periodic(coarse_truth, radius=radius)
    pats_c = P.pattern_array(seen_c, radius=radius)

    rng = np.random.default_rng(seed)
    coarse_seed = (rng.random(coarse_truth.shape) < 0.5).astype(int)  # pure static
    coarse_grown = P.run_sync(coarse_seed, seen_c, pats_c,
                              mode="hybrid", radius=radius,
                              sweeps=sweeps, verbose=verbose)

    # --- fine stage: upsample, de-block, refine with the fine rule ---
    seed_fine = upsample(coarse_grown, k, out=n)
    seed_fine = smooth_binary(seed_fine, scale=smooth_scale, frac=frac)

    seen_f = P.build_windows_periodic(truth, radius=radius)
    pats_f = P.pattern_array(seen_f, radius=radius)
    grown = P.run_sync(seed_fine, seen_f, pats_f,
                       mode="hybrid", radius=radius,
                       sweeps=sweeps, verbose=verbose)
    grown = P.clean_specks(grown, thresh=7, rounds=3)
    return grown