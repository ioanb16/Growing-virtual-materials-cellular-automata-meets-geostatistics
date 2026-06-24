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