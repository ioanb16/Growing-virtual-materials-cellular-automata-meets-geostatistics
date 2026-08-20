"""
pattern_ca.py
=============
Probability tables (pattern vocabulary) and the hybrid cellular-automaton
repair rule, with periodic (wrap-around) boundaries throughout.

MODULAR NEIGHBOURHOOD: every function takes `radius` (default 1).
    radius=1 -> 3x3 window (9 cells)      <- the original behaviour
    radius=2 -> 5x5 window (25 cells)
    radius=k -> (2k+1)x(2k+1) window
The centre cell is always the one decided/repaired; only the window size changes.

Pipeline (pick a radius and keep it the same everywhere):
    R = 2
    seen  = build_windows_periodic(truth, radius=R)   # legal vocabulary (a set)
    table = build_pattern_table(truth,     radius=R)  # same, WITH counts
    P     = pattern_array(seen, radius=R)             # patterns as (P, w, w) array
    out   = run_sync(noisy, seen, P, radius=R)        # repair with hybrid rule
    out   = clean_specks(out)                          # 3x3 majority cleanup (fixed)
    out   = remove_small_blobs(out, min_size=4)        # optional: connectivity cleanup

Two cleanups, used in order and for different jobs:
    clean_specks       local 3x3 majority; removes specks nearly SURROUNDED by
                       the other phase. Fast, safe, but can't touch a speck that
                       sits on a real edge (it has same-colour neighbours).
    remove_small_blobs labels whole connected blobs (periodic) and drops any
                       below min_size, so it clears edge-hugging specks the
                       majority filter leaves. Runs in <0.1s; use on the real
                       rock where clean_specks alone leaves salt-and-pepper.

NOTE on radius and the real rock: widening the window (radius=2) helps rigid,
tiny-vocabulary structures (chessboard) but HURTS the sandstone - its vocabulary
explodes to tens of thousands of patterns, so specks hide as "legal" windows AND
every cell's nearest-pattern scan slows to minutes. Keep the rock at radius=1 and
lean on the two cleanups above.

The hybrid rule for each cell (synchronous, periodic window):
    1. window recognised            -> leave it
    2. flipping the centre makes it legal -> flip the centre
    3. otherwise                    -> set centre to that of nearest legal pattern

Dependencies: numpy, scipy   (collections.Counter from stdlib)

NOTE (honest limitation & why radius matters):
A small window can't tell a noise speck from a genuine thin feature, because the
speck's window matches a legal thin-feature pattern and is kept. Enlarging the
radius gives the centre more context (a 5x5 sees 24 neighbours vs a 3x3's 8), so
isolated specks are more likely to be seen as illegal and repaired. The cost is a
much larger vocabulary: 3x3 has 2^9=512 possible patterns, 5x5 has 2^25~=33M, so
learning is heavier and nearest_pattern (a scan over all stored patterns) slows.
"""

import numpy as np
from collections import Counter
from scipy import ndimage as ndi


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _win_size(radius):
    """Side length of the window for a given radius (3x3 for r=1, 5x5 for r=2)."""
    return 2 * radius + 1


def win_key(w):
    """All cells of a window, row-major, as a hashable tuple.
    Works for any square window size."""
    return tuple(int(x) for x in w.ravel())


# ---------------------------------------------------------------------------
# Vocabulary / probability tables
# ---------------------------------------------------------------------------
def build_windows_periodic(grid, radius=1):
    """The SET of every (2r+1)x(2r+1) pattern in the grid, read with wrap-around.
    Covers all cells (no untouchable border)."""
    w = _win_size(radius)
    pad = np.pad(grid, radius, mode="wrap")
    seen = set()
    rows, cols = grid.shape
    for r in range(rows):
        for c in range(cols):
            seen.add(win_key(pad[r:r + w, c:c + w]))
    return seen


def build_pattern_table(grid, radius=1):
    """The probability table: every (2r+1)x(2r+1) pattern WITH its count,
    read with wrap-around (vectorised). Returns a collections.Counter
    {pattern_key: count}. Divide by the total for probabilities."""
    w = _win_size(radius)
    pad = np.pad(grid, radius, mode="wrap")
    rows, cols = grid.shape
    keys = np.empty((rows * cols, w * w), dtype=np.int8)
    k = 0
    for dr in range(w):
        for dc in range(w):
            keys[:, k] = pad[dr:dr + rows, dc:dc + cols].ravel()
            k += 1
    return Counter(map(tuple, keys))


def pattern_array(seen, radius=1):
    """Recognised patterns as a (P, w, w) int array, in sorted order.
    Accepts a set OR a Counter (uses its keys)."""
    w = _win_size(radius)
    return np.array([np.array(p, dtype=int).reshape(w, w) for p in sorted(seen)])


# ---------------------------------------------------------------------------
# The hybrid repair rule (synchronous, periodic, any radius)
# ---------------------------------------------------------------------------
def nearest_pattern(win, patterns):
    """Recognised pattern closest to window `win` by Hamming distance.
    Ties: change the centre as little as possible, then lowest sorted order.
    `patterns` is (P, w, w); centre index is w//2 in each axis."""
    ctr = patterns.shape[1] // 2
    diffs = (patterns != win).reshape(len(patterns), -1).sum(axis=1)
    cands = np.where(diffs == diffs.min())[0]
    if len(cands) > 1:
        centre_change = (patterns[cands, ctr, ctr] != win[ctr, ctr]).astype(int)
        cands = cands[centre_change == centre_change.min()]
    return patterns[cands[0]]


def sweep_sync(field, seen, patterns, mode="hybrid", radius=1):
    """One synchronous sweep, periodic boundaries. Every cell reads a frozen
    snapshot and writes only its own (centre) value.
      mode='center' -> flip-centre only
      mode='hybrid' -> flip-centre, then nearest-pattern fallback
    Returns (next_field, n_cells_changed).
    """
    w = _win_size(radius)
    ctr = radius
    rows, cols = field.shape
    pad = np.pad(field, radius, mode="wrap")
    nxt = field.copy()
    for r in range(rows):
        for c in range(cols):
            win = pad[r:r + w, c:c + w]
            if win_key(win) in seen:
                continue
            trial = win.copy()
            trial[ctr, ctr] ^= 1                     # try flipping just the centre
            if win_key(trial) in seen:
                nxt[r, c] = trial[ctr, ctr]
                continue
            if mode == "hybrid":                     # fallback: centre of nearest pattern
                nxt[r, c] = int(nearest_pattern(win, patterns)[ctr, ctr])
    return nxt, int((nxt != field).sum())


def run_sync(field, seen, patterns, mode="hybrid", radius=1, sweeps=100, verbose=True):
    """Repeat synchronous sweeps until a fixed point or 2-cycle, or the cap.
    Input `field` is never modified. Keep `radius` the same as used to learn
    `seen`/`patterns`."""
    f = field.copy()
    prev2 = None
    for t in range(1, sweeps + 1):
        nxt, changed = sweep_sync(f, seen, patterns, mode, radius)
        if changed == 0:
            if verbose:
                print(f"fixed point at sweep {t}")
            return nxt
        if prev2 is not None and np.array_equal(nxt, prev2):
            if verbose:
                print(f"2-cycle at sweep {t}")
            return nxt
        prev2 = f
        f = nxt
    if verbose:
        print("hit sweep cap")
    return f


# ---------------------------------------------------------------------------
# SEQUENTIAL engine (in-place, random visit order, whole-window stamping)
# ---------------------------------------------------------------------------
# Difference from the synchronous engine above:
#   - cells are visited one at a time in random order and updated IN PLACE, so a
#     later cell sees earlier cells' updates within the same sweep;
#   - the nearest-pattern fallback STAMPS THE WHOLE WINDOW, not just the centre.
# This whole-window stamping lets a neighbour reach in and clear specks that the
# synchronous (centre-only) engine leaves behind, so it repairs fine patterns
# (e.g. the chessboard) far better. Trade-off (documented): it can drift a phase
# interface, because it overwrites already-correct cells. Uses periodic wrap.
def step_hybrid(field, seen, patterns, rng, radius=1, mode="hybrid"):
    """One sequential sweep, periodic. Visit interior+wrapped cells in random
    order; recognise->leave, flip-centre->check, else fallback.
    Fallback depends on `mode`:
      'hybrid'   -> stamp the whole nearest legal pattern (Hamming distance)
      'majority' -> set the centre to the majority phase of its window
    Modifies `field` in place. Returns (field, cells_changed)."""
    w = _win_size(radius)
    ctr = radius
    rows, cols = field.shape
    sites = [(r, c) for r in range(rows) for c in range(cols)]
    changed = 0
    for i in rng.permutation(len(sites)):
        r, c = sites[i]
        rr = [(r + dr - radius) % rows for dr in range(w)]
        cc = [(c + dc - radius) % cols for dc in range(w)]
        win = field[np.ix_(rr, cc)]

        if win_key(win) in seen:
            continue

        trial = win.copy()
        trial[ctr, ctr] ^= 1
        if win_key(trial) in seen:
            field[r, c] = trial[ctr, ctr]
            changed += 1
            continue

        if mode == "majority":
            # set the centre to whichever phase is the majority in the window
            new_centre = int(win.mean() >= 0.5)
            if new_centre != win[ctr, ctr]:
                field[r, c] = new_centre
                changed += 1
        else:  # 'hybrid': whole-window stamp of nearest legal pattern
            new = nearest_pattern(win, patterns)
            if not np.array_equal(new, win):
                changed += int((new != win).sum())
                field[np.ix_(rr, cc)] = new
    return field, changed


def run_hybrid(field, seen, patterns, radius=1, sweeps=100, seed=0, mode="hybrid", verbose=True):
    """Repeat sequential sweeps until nothing changes or the cap. Input `field`
    is never modified. Keep `radius` the same as used to learn seen/patterns.
      mode='hybrid'   -> nearest-pattern (Hamming) whole-window fallback
      mode='majority' -> majority-vote of the window on the centre cell
    NOTE: on symmetric patterns (e.g. chessboard) heavy noise can converge to the
    phase-INVERTED image, which is still a perfect reconstruction - compare
    structure, not raw pixel-difference-from-original."""
    rng = np.random.default_rng(seed)
    f = field.copy()
    prev = None
    for t in range(1, sweeps + 1):
        f, changed = step_hybrid(f, seen, patterns, rng, radius, mode)
        if verbose and changed != prev:
            print(f"  sweep {t:5d}: cells changed = {changed}")
            prev = changed
        if changed == 0:
            break
    return f


# ---------------------------------------------------------------------------
# Speck cleanup (periodic 8-neighbour majority filter; independent of radius)
# ---------------------------------------------------------------------------
def clean_specks(field, thresh=7, rounds=3):
    """Flip a cell only if >= thresh of its 8 periodic neighbours disagree.
    thresh=7/8 removes near-surrounded specks & pits while leaving real edges
    untouched. thresh<=6 starts eroding corners/thin features.
    (This is a fixed 3x3 majority filter, run after the CA regardless of radius.)"""
    f = field.copy()
    for _ in range(rounds):
        pad = np.pad(f, 1, mode="wrap")
        white_nb = (pad[:-2, :-2] + pad[:-2, 1:-1] + pad[:-2, 2:] +
                    pad[1:-1, :-2]                  + pad[1:-1, 2:] +
                    pad[2:, :-2]  + pad[2:, 1:-1]   + pad[2:, 2:])
        black_nb = 8 - white_nb
        flip_white = (f == 1) & (black_nb >= thresh)
        flip_black = (f == 0) & (white_nb >= thresh)
        f = np.where(flip_white, 0, np.where(flip_black, 1, f))
    return f


# ---------------------------------------------------------------------------
# Connectivity cleanup (periodic; removes small blobs the majority filter can't)
# ---------------------------------------------------------------------------
def remove_small_blobs(field, min_size=4, connectivity=1):
    """Delete connected components smaller than `min_size` cells, in BOTH phases,
    with periodic (wrap-around) connectivity.

    Difference from clean_specks: that is a local 3x3 majority vote, so it only
    removes specks that are nearly SURROUNDED by the other phase. A speck sitting
    ON a real edge has same-colour neighbours (the edge), so the majority filter
    leaves it. This function instead labels whole connected blobs and removes any
    blob below `min_size` regardless of where it sits, so it clears edge-hugging
    specks WITHOUT eroding the edge itself.

      min_size     : blobs with fewer than this many cells are removed (flipped
                     to the other phase). min_size=4 clears 1-3 px specks.
      connectivity : 1 -> 4-neighbour (orthogonal), 2 -> 8-neighbour (incl. diag).

    Periodicity is handled by tiling 2x2, labelling, then reading back the centre
    tile, so blobs that wrap across an edge count as one. Returns an int 0/1 array.
    """
    f = (np.asarray(field) > 0).astype(int)
    ny, nx = f.shape
    struct = ndi.generate_binary_structure(2, connectivity)

    for phase in (1, 0):
        mask = (f == phase)
        # tile 2x2 so wrap-around blobs are connected, label, crop centre tile
        tiled = np.tile(mask, (2, 2))
        lab, n = ndi.label(tiled, structure=struct)
        if n == 0:
            continue
        sizes = np.bincount(lab.ravel())
        lab_centre = lab[:ny, :nx]
        # size of each cell's blob (measured on the full tiling)
        cell_size = sizes[lab_centre]
        small = (lab_centre > 0) & (cell_size < min_size)
        f = np.where(small, 1 - phase, f)
    return f