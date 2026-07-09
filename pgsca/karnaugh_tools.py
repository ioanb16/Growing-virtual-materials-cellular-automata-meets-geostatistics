import numpy as np
from scipy.ndimage import label as _label

def encode_neighbourhood(grid, i, j):
    """
    Returns an integer 0-80 uniquely identifying the 4-neighbour (von Neumann) 
    pattern (N, S, E, W) using a base-3 encoding system.
    """
    rows, cols = grid.shape
    n = int(grid[i-1, j]) if i > 0        else 0
    s = int(grid[i+1, j]) if i < rows - 1 else 0
    e = int(grid[i, j+1]) if j < cols - 1 else 0
    w = int(grid[i, j-1]) if j > 0        else 0

    # Base-3 encoding for 3 possible states (0, 1, 2)
    return n * 27 + s * 9 + e * 3 + w


def build_table(pairs, neighbourhood='von_neumann'):
    """
    Builds a probability table from training data.
    table[pattern_id, state] = P(output state | neighbourhood pattern)
    """
    n_patterns = 6561 if neighbourhood == 'moore' else 81
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
            pattern_ids = (n*2187 + s*729 + e*243 + w*81 + ne*27 + nw*9 + se*3 + sw)
        else:
            pattern_ids = n*27 + s*9 + e*3 + w

        linear_idx = pattern_ids.ravel() * 3 + target.ravel().astype(int)
        count += np.bincount(linear_idx, minlength=n_patterns*3).reshape(n_patterns, 3)

    row_totals = count.sum(axis=1, keepdims=True)
    row_totals[row_totals == 0] = 1
    return count / row_totals


def apply_table(pgs, table, rng=None):
    """Applies the Karnaugh map table to every cell in one vectorised pass."""
    if rng is None:
        rng = np.random.default_rng()

    use_moore = (table.shape[0] == 6561)
    rows, cols = pgs.shape
    padded = np.pad(pgs, pad_width=1, mode='constant', constant_values=0)

    n  = padded[0:rows,   1:cols+1].astype(int)
    s  = padded[2:rows+2, 1:cols+1].astype(int)
    e  = padded[1:rows+1, 2:cols+2].astype(int)
    w  = padded[1:rows+1, 0:cols  ].astype(int)

    if use_moore:
        ne = padded[0:rows,   2:cols+2].astype(int)
        nw = padded[0:rows,   0:cols  ].astype(int)
        se = padded[2:rows+2, 2:cols+2].astype(int)
        sw = padded[2:rows+2, 0:cols  ].astype(int)
        pattern_ids = (n*2187 + s*729 + e*243 + w*81 + ne*27 + nw*9 + se*3 + sw)
    else:
        pattern_ids = n*27 + s*9 + e*3 + w

    probs    = table[pattern_ids.ravel()]
    cumprobs = np.cumsum(probs, axis=1)
    r        = rng.random(rows * cols)[:, np.newaxis]
    output   = (r > cumprobs).sum(axis=1).reshape(rows, cols).astype(int)

    unseen = (probs.sum(axis=1) == 0).reshape(rows, cols)
    output[unseen] = pgs[unseen].astype(int)

    return output


def sequential_simulate(table, shape, proportions=None, n_passes=10,
                        improvement_tol=0.005, rng=None,
                        initial_grid=None, track_phase=None):
    """
    Iteratively refines a grid using Sequential Indicator Simulation principles.
    Note: For very large grids, wrapping the inner loop with Numba @njit is recommended.
    """
    if rng is None:
        rng = np.random.default_rng()
    if proportions is None:
        proportions = [1/3, 1/3, 1/3]

    use_moore = (table.shape[0] == 6561)
    target_props = np.array(proportions, dtype=float)
    rows, cols = shape

    if initial_grid is not None:
        if initial_grid.shape != (rows, cols):
            raise ValueError(f"initial_grid shape {initial_grid.shape} does not match requested shape {(rows, cols)}")
        grid = initial_grid.astype(int).copy()
    else:
        grid = rng.choice(3, size=(rows, cols), p=target_props)

    history = []
    trace   = []
    if track_phase is not None:
        trace.append(compute_morphology(grid)[track_phase])

    for pass_num in range(n_passes):
        prev = grid.copy()

        counts = np.bincount(grid.ravel(), minlength=3).astype(float)
        current_props = counts / counts.sum()

        order = np.arange(rows * cols)
        rng.shuffle(order)

        for idx in order:
            i, j = divmod(idx, cols)

            n = int(grid[i-1, j]) if i > 0        else 0
            s = int(grid[i+1, j]) if i < rows - 1 else 0
            e = int(grid[i, j+1]) if j < cols - 1 else 0
            w = int(grid[i, j-1]) if j > 0        else 0

            if use_moore:
                ne = int(grid[i-1, j+1]) if i > 0 and j < cols-1     else 0
                nw = int(grid[i-1, j-1]) if i > 0 and j > 0           else 0
                se = int(grid[i+1, j+1]) if i < rows-1 and j < cols-1 else 0
                sw = int(grid[i+1, j-1]) if i < rows-1 and j > 0      else 0
                pattern_id = (n*2187 + s*729 + e*243 + w*81 + ne*27 + nw*9 + se*3 + sw)
            else:
                pattern_id = n*27 + s*9 + e*3 + w

            # Smooth correction avoids astronomical multiplier spikes if a phase hits 0
            correction = target_props / (current_props + 1e-5)
            probs      = table[pattern_id] * correction
            total      = probs.sum()
            probs      = target_props.copy() if total == 0 else probs / total

            new_state = rng.choice(3, p=probs)

            current_props[grid[i, j]] -= 1 / (rows * cols)
            current_props[new_state]  += 1 / (rows * cols)
            current_props              = np.clip(current_props, 0, None)

            grid[i, j] = new_state

        delta = np.mean(grid != prev)
        history.append(delta)
        if track_phase is not None:
            trace.append(compute_morphology(grid)[track_phase])

        if pass_num > 0 and (history[-2] - history[-1]) < improvement_tol:
            improvement = history[-2] - history[-1]
            label = "Converged" if improvement >= 0 else "Stopped (no further improvement)"
            print(f"{label} after {pass_num + 1} pass(es)  (improvement = {improvement:.4f} < {improvement_tol})")
            break
    else:
        print(f"Reached max passes ({n_passes})  (final δ = {history[-1]:.4f})")

    if track_phase is not None:
        return grid, history, trace
    return grid, history


def compute_morphology(grid):
    """
    Computes spatial metrics for the phases in the grid.
    Returns a dict keyed by phase (0, 1, 2) containing proportions, areas, and percolation state.
    """
    metrics = {}
    rows, cols = grid.shape

    for phase in range(3):
        mask       = (grid == phase)
        labeled, n = _label(mask)

        if n == 0:
            metrics[phase] = dict(proportion=0.0, n_components=0, mean_area=0.0, 
                                  largest_area=0, mean_diameter=0.0, percolates=False)
            continue

        areas = np.bincount(labeled.ravel())[1:]

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