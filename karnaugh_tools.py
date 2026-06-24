import numpy as np

def encode_neighbourhood(grid, i, j, help=False):
    if help:
        print("""
        encode_neighbourhood() parameters:
        grid : 2D numpy array of integer states (0, 1, or 2)
        i    : row index of the centre cell
        j    : column index of the centre cell

        Returns an integer 0-80 uniquely identifying the
        4-neighbour pattern (N, S, E, W) using fixed boundaries
        (missing neighbours treated as state 0).
        """)
        return

    rows, cols = grid.shape
    n = int(grid[i-1, j]) if i > 0          else 0
    s = int(grid[i+1, j]) if i < rows - 1   else 0
    e = int(grid[i, j+1]) if j < cols - 1   else 0
    w = int(grid[i, j-1]) if j > 0          else 0

    return n * 27 + s * 9 + e * 3 + w



def build_table(pairs, help=False):
    if help:
        print("""
        build_table() parameters:
        pairs : list of (pgs_field, target_field) tuples
                both must be 2D numpy arrays of integers 0, 1, or 2
                and the same shape

        Returns a (81, 3) numpy array where
        table[pattern_id, state] = P(output state | neighbourhood pattern)
        """)
        return

    count = np.zeros((81, 3), dtype=int)

    for pgs, target in pairs:
        rows, cols = pgs.shape
        for i in range(rows):
            for j in range(cols):
                pattern_id = encode_neighbourhood(pgs, i, j)
                output_state = int(target[i, j])
                count[pattern_id, output_state] += 1

    # avoid division by zero for patterns that never appeared
    row_totals = count.sum(axis=1, keepdims=True)
    row_totals[row_totals == 0] = 1

    table = count / row_totals
    return table