
import numpy as np
import gstools as gs
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib.patches import Patch
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

def get_neighbours(grid, i, j, neighbourhood='von_neumann', radius=1, boundary='fixed',
                    weights=None, direct_weight=1, diagonal_weight=1):
    rows, cols = grid.shape
    values = []
    value_weights = []

    for di in range(-radius, radius + 1):
        for dj in range(-radius, radius + 1):
            if di == 0 and dj == 0:
                continue

            if neighbourhood == 'von_neumann':
                if abs(di) + abs(dj) > radius:
                    continue
            elif neighbourhood == 'moore':
                if max(abs(di), abs(dj)) > radius:
                    continue

            ni, nj = i + di, j + dj

            if boundary == 'fixed':
                if ni < 0 or ni >= rows or nj < 0 or nj >= cols:
                    continue
            elif boundary == 'periodic':
                ni %= rows
                nj %= cols
            elif boundary == 'reflective':
                if ni < 0:
                    ni = -ni
                elif ni >= rows:
                    ni = 2 * (rows - 1) - ni
                if nj < 0:
                    nj = -nj
                elif nj >= cols:
                    nj = 2 * (cols - 1) - nj

            values.append(grid[ni, nj])

            # weight lookup: custom dict takes priority, otherwise direct/diagonal default
            if weights is not None and (di, dj) in weights:
                value_weights.append(weights[(di, dj)])
            else:
                value_weights.append(direct_weight if (di == 0 or dj == 0) else diagonal_weight)

    return values, value_weights

def run_ca(
    lithotype_map,
    generations=100, threshold=1, checkpoints=[10, 100],
    help=False
):
    if help:
        print("""
        run_ca() parameters:
        lithotype_map : the map to evolve (from make_lithotype_map)
        generations   : number of CA generations (default 100)
        threshold     : neighbour agreement needed to change a cell (default 1)
        checkpoints   : generations to print stats at (default [10, 100])
              
        returns:
        lithotype_map : the evolved map after the specified generations
        snapshots     : dict of {generation: map_copy} for each checkpoint
        """)
        return
    
    snapshots = {}

    for generation in range(generations):
        new_grid = lithotype_map.copy()
        for i in range(1, lithotype_map.shape[0]-1):
            for j in range(1, lithotype_map.shape[1]-1):
                neighbours = [lithotype_map[i-1, j], lithotype_map[i+1, j],
                              lithotype_map[i, j-1], lithotype_map[i, j+1]]
                counts = np.bincount(neighbours, minlength=3)
                if counts.max() > threshold:
                    new_grid[i, j] = counts.argmax()
        lithotype_map = new_grid
        if generation+1 in checkpoints:
            total = lithotype_map.size
            print(f"Generation {generation+1}:")
            print(f"  Rock type 0: {(lithotype_map == 0).sum() / total * 100:.1f}%")
            print(f"  Rock type 1: {(lithotype_map == 1).sum() / total * 100:.1f}%")
            print(f"  Rock type 2: {(lithotype_map == 2).sum() / total * 100:.1f}%")
            snapshots[generation+1] = lithotype_map.copy()
    
    return lithotype_map, snapshots

def plot_ca_evolution(
    snapshots,
    cmap='viridis', figsize=(6, 6), interval=500,
    help=False
):
    if help:
        print("""
        plot_ca_evolution() parameters:
        snapshots : dict of {generation: map} from run_ca
        cmap      : colormap (default 'viridis')
        figsize   : figure size (default (6, 6))
        interval  : milliseconds between frames (default 500)
        """)
        return

    generations = sorted(snapshots.keys())
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(snapshots[generations[0]], cmap=cmap, vmin=0, vmax=2)
    title = ax.set_title(f"Generation {generations[0]}")

    def update(frame):
        gen = generations[frame]
        im.set_data(snapshots[gen])
        title.set_text(f"Generation {gen}")
        return im, title

    ani = FuncAnimation(fig, update, frames=len(generations), interval=interval)
    plt.close(fig)
    return HTML(ani.to_jshtml())



