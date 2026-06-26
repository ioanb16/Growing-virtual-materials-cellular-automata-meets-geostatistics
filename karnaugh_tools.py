import numpy as np
from scipy.ndimage import label as _label

def encode_neighbourhood(grid, i, j, help=False):
    if help:
        print("""
        encode_neighbourhood() parameters:
        grid : 2D numpy array of integer states (0, 1, or 2)
        i    : row index of the centre cell
        j    : column index of the centre cell

        Returns an integer 0-80 uniquely identifying the
        4-neighbour (von Neumann) pattern (N, S, E, W) using
        fixed boundaries (missing neighbours treated as state 0).
        """)
        return

    rows, cols = grid.shape
    n = int(grid[i-1, j]) if i > 0        else 0
    s = int(grid[i+1, j]) if i < rows - 1 else 0
    e = int(grid[i, j+1]) if j < cols - 1 else 0
    w = int(grid[i, j-1]) if j > 0        else 0

    return n * 27 + s * 9 + e * 3 + w


def build_table(pairs, neighbourhood='von_neumann', help=False):
    if help:
        print("""
        build_table() parameters:
        pairs        : list of (pgs_field, target_field) tuples
                       both must be 2D numpy arrays of integers 0, 1, or 2
                       and the same shape
        neighbourhood: 'von_neumann' (4 neighbours, 81 patterns)  [default]
                       'moore'       (8 neighbours, 6561 patterns)

        Returns a (81, 3) or (6561, 3) numpy array where
        table[pattern_id, state] = P(output state | neighbourhood pattern)
        """)
        return

    if neighbourhood == 'moore':
        n_patterns = 6561
    else:
        n_patterns = 81

    count = np.zeros((n_patterns, 3), dtype=int)

    for pgs, target in pairs:
        rows, cols = pgs.shape
        padded = np.pad(pgs, pad_width=1, mode='constant', constant_values=0)

        n  = padded[0:rows,   1:cols+1].astype(int)
        s  = padded[2:rows+2, 1:cols+1].astype(int)
        e  = padded[1:rows+1, 2:cols+2].astype(int)
        w  = padded[1:rows+1, 0:cols  ].astype(int)

        if neighbourhood == 'moore':
            ne = padded[0:rows,   2:cols+2].astype(int)
            nw = padded[0:rows,   0:cols  ].astype(int)
            se = padded[2:rows+2, 2:cols+2].astype(int)
            sw = padded[2:rows+2, 0:cols  ].astype(int)
            pattern_ids = (n*2187 + s*729 + e*243 + w*81 +
                           ne*27  + nw*9  + se*3  + sw)
        else:
            pattern_ids = n*27 + s*9 + e*3 + w

        linear_idx = pattern_ids.ravel() * 3 + target.ravel().astype(int)
        count += np.bincount(linear_idx, minlength=n_patterns*3).reshape(n_patterns, 3)

    row_totals = count.sum(axis=1, keepdims=True)
    row_totals[row_totals == 0] = 1
    table = count / row_totals
    return table


def apply_table(pgs, table, rng=None, help=False):
    if help:
        print("""
        apply_table() parameters:
        pgs   : 2D numpy array of integers 0, 1, 2 (the input PGS field)
        table : (81, 3) or (6561, 3) probability array from build_table()
        rng   : numpy random Generator for reproducibility (optional)

        Returns a 2D array the same shape as pgs where each cell
        has been assigned a state sampled from the table row
        corresponding to its neighbourhood pattern.
        """)
        return

    if rng is None:
        rng = np.random.default_rng()

    rows, cols = pgs.shape
    output = np.zeros((rows, cols), dtype=int)

    for i in range(rows):
        for j in range(cols):
            pattern_id = encode_neighbourhood(pgs, i, j)
            probs = table[pattern_id]
            output[i, j] = rng.choice(3, p=probs)

    return output


def sequential_simulate(table, shape, proportions=None, n_passes=10, improvement_tol=0.005, rng=None, help=False):
    if help:
        print("""
        sequential_simulate() parameters:
        table           : (81, 3) or (6561, 3) probability array from build_table()
                          neighbourhood type is inferred from table shape
        shape           : tuple (rows, cols) for the output grid
        proportions     : list [p0, p1, p2] — used both for random
                          initialisation AND as proportion conditioning
                          targets during sampling (default: uniform)
        n_passes        : maximum number of refinement passes (default 10)
        improvement_tol : stop early if the drop in change-fraction between
                          consecutive passes falls below this value (default 0.005)
        rng             : numpy random Generator (optional)

        Returns (grid, history) where
          grid    : final 2D array of shape `shape`
          history : list of per-pass change fractions (length = passes run)
        """)
        return

    if rng is None:
        rng = np.random.default_rng()
    if proportions is None:
        proportions = [1/3, 1/3, 1/3]

    use_moore = (table.shape[0] == 6561)
    target_props = np.array(proportions, dtype=float)
    rows, cols = shape
    grid = rng.choice(3, size=(rows, cols), p=target_props)

    history = []

    for pass_num in range(n_passes):
        prev = grid.copy()

        counts = np.bincount(grid.ravel(), minlength=3).astype(float)
        current_props = counts / counts.sum()

        order = [(i, j) for i in range(rows) for j in range(cols)]
        rng.shuffle(order)

        for i, j in order:
            n = int(grid[i-1, j]) if i > 0        else 0
            s = int(grid[i+1, j]) if i < rows - 1 else 0
            e = int(grid[i, j+1]) if j < cols - 1 else 0
            w = int(grid[i, j-1]) if j > 0        else 0

            if use_moore:
                ne = int(grid[i-1, j+1]) if i > 0 and j < cols-1      else 0
                nw = int(grid[i-1, j-1]) if i > 0 and j > 0            else 0
                se = int(grid[i+1, j+1]) if i < rows-1 and j < cols-1  else 0
                sw = int(grid[i+1, j-1]) if i < rows-1 and j > 0       else 0
                pattern_id = (n*2187 + s*729 + e*243 + w*81 +
                              ne*27  + nw*9  + se*3  + sw)
            else:
                pattern_id = n*27 + s*9 + e*3 + w

            correction = target_props / np.clip(current_props, 1e-6, None)
            probs = table[pattern_id] * correction
            total = probs.sum()
            if total == 0:
                probs = target_props.copy()
            else:
                probs = probs / total

            new_state = rng.choice(3, p=probs)

            current_props[grid[i, j]] -= 1 / (rows * cols)
            current_props[new_state]   += 1 / (rows * cols)
            current_props = np.clip(current_props, 0, None)

            grid[i, j] = new_state

        delta = np.mean(grid != prev)
        history.append(delta)

        if pass_num > 0 and (history[-2] - history[-1]) < improvement_tol:
            print(f"Converged after {pass_num + 1} pass(es)  (improvement = {history[-2]-history[-1]:.4f} < {improvement_tol})")
            break
    else:
        print(f"Reached max passes ({n_passes})  (final δ = {history[-1]:.4f})")

    return grid, history



def compute_morphology(grid, help=False):
    if help:
        print("""
        compute_morphology() parameters:
        grid : 2D numpy array of integers 0, 1, 2

        Returns a dict keyed by phase (0, 1, 2), each containing:
            proportion    : fraction of grid occupied by this phase
            n_components  : number of connected components (blobs)
            mean_area     : mean blob area in pixels
            largest_area  : largest blob area in pixels
            mean_diameter : mean equivalent circular diameter (pixels)
                            d = 2 * sqrt(area / pi)
            percolates    : bool — does any blob span top row to bottom row?
        """)
        return

    metrics = {}
    rows, cols = grid.shape

    for phase in range(3):
        mask = (grid == phase)
        labeled, n = _label(mask)

        if n == 0:
            metrics[phase] = dict(proportion=0.0, n_components=0,
                                  mean_area=0.0, largest_area=0,
                                  mean_diameter=0.0, percolates=False)
            continue

        areas = np.array([np.sum(labeled == k) for k in range(1, n + 1)])

        percolates = any(
            (labeled == k)[0, :].any() and (labeled == k)[-1, :].any()
            for k in range(1, n + 1)
        )

        mean_area = areas.mean()

        metrics[phase] = dict(
            proportion    = float(mask.mean()),
            n_components  = n,
            mean_area     = float(mean_area),
            largest_area  = int(areas.max()),
            mean_diameter = float(2 * np.sqrt(mean_area / np.pi)),
            percolates    = percolates,
        )

    return metrics