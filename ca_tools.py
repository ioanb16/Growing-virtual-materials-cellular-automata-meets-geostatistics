
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


def decide_new_state(neighbour_values, current_state, threshold=1, neighbour_weights=None,
                      rule='majority', temperature=0.0, interaction_matrix=None, rng=None):
    counts = np.bincount(neighbour_values, weights=neighbour_weights, minlength=3)

    if rule == 'majority':
        majority_state = counts.argmax()
        eff_threshold = threshold[current_state] if isinstance(threshold, dict) else threshold
        if counts.max() > eff_threshold:
            return majority_state
        return current_state

    elif rule == 'probabilistic':
        if interaction_matrix is None:
            # default: free to touch your own type, costs 1 to touch any different type
            interaction_matrix = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        if rng is None:
            rng = np.random.default_rng()

        candidate_state = counts.argmax()
        if candidate_state == current_state:
            return current_state

        n_states = len(counts)
        energy_current = sum(interaction_matrix[current_state, s] * counts[s] for s in range(n_states))
        energy_candidate = sum(interaction_matrix[candidate_state, s] * counts[s] for s in range(n_states))
        delta = energy_candidate - energy_current

        if delta <= 0:
            return candidate_state   # switching lowers energy - always favourable, just do it
        if temperature <= 0:
            return current_state     # switching raises energy, and no randomness allowed - refuse

        probability = np.exp(-delta / temperature)
        if rng.random() < probability:
            return candidate_state   # accept the unfavourable switch anyway, by chance
        return current_state

def run_ca(
    lithotype_map,
    generations=100, checkpoints=[10, 100],
    neighbourhood='von_neumann', radius=1, boundary='fixed',
    weights=None, direct_weight=1, diagonal_weight=1,
    rule='majority', threshold=1, temperature=0.0, interaction_matrix=None, rng=None,
    update_scheme='synchronous',
    locked_mask=None,
    nucleation_rate=0.0,
    target_proportions=None, conservation_strength=0.0, conservation_scale=10,
    help=False
):
    if help:
        print("""
        run_ca() parameters:
        lithotype_map   : the map to evolve (from make_lithotype_map)
        generations     : number of CA generations (default 100)
        checkpoints     : generations to print stats at (default [10, 100])

        neighbourhood, radius, boundary, weights, direct_weight, diagonal_weight
                        : passed straight through to get_neighbours() - see its help

        rule, threshold, temperature, interaction_matrix
                        : passed straight through to decide_new_state() - see its help

        rng             : numpy random Generator, for reproducible probabilistic/async/nucleation behaviour

        update_scheme   : 'synchronous' or 'asynchronous' - see stage 3 notes

        locked_mask     : boolean array same shape as the map. True = that cell never changes

        nucleation_rate : probability per cell per generation of spontaneously becoming
                          a random state, regardless of neighbours (default 0.0 = off)

        target_proportions    : dict {state: proportion} to pull the map back toward
        conservation_strength : how strongly to enforce target_proportions (default 0.0 = off)
        conservation_scale    : multiplier on the threshold adjustment (default 10).
                                NOTE: this whole mechanism is an untested heuristic -
                                tested once and didn't clearly work with the majority
                                rule's integer counts. Needs real tuning later, against
                                actual morphology metrics, not guessed blindly now.

        returns:
        lithotype_map : the evolved map after the specified generations
        snapshots     : dict of {generation: map_copy} for each checkpoint
        """)
        return

    if rng is None:
        rng = np.random.default_rng()

    snapshots = {}
    rows, cols = lithotype_map.shape

    for generation in range(generations):

        if target_proportions is not None and conservation_strength > 0:
            counts = np.bincount(lithotype_map.flatten(), minlength=3)
            current_proportions = counts / lithotype_map.size
            effective_threshold = {}
            for state in range(3):
                base = threshold[state] if isinstance(threshold, dict) else threshold
                target = target_proportions.get(state, current_proportions[state])
                drift = target - current_proportions[state]
                effective_threshold[state] = base + conservation_strength * drift * conservation_scale
        else:
            effective_threshold = threshold

        if update_scheme == 'synchronous':
            new_grid = lithotype_map.copy()
            for i in range(rows):
                for j in range(cols):
                    if locked_mask is not None and locked_mask[i, j]:
                        continue
                    values, w = get_neighbours(lithotype_map, i, j,
                                                neighbourhood=neighbourhood, radius=radius, boundary=boundary,
                                                weights=weights, direct_weight=direct_weight, diagonal_weight=diagonal_weight)
                    new_grid[i, j] = decide_new_state(values, lithotype_map[i, j], threshold=effective_threshold,
                                                       neighbour_weights=w, rule=rule, temperature=temperature,
                                                       interaction_matrix=interaction_matrix, rng=rng)
            lithotype_map = new_grid

        elif update_scheme == 'asynchronous':
            order = [(i, j) for i in range(rows) for j in range(cols)]
            rng.shuffle(order)
            for i, j in order:
                if locked_mask is not None and locked_mask[i, j]:
                    continue
                values, w = get_neighbours(lithotype_map, i, j,
                                            neighbourhood=neighbourhood, radius=radius, boundary=boundary,
                                            weights=weights, direct_weight=direct_weight, diagonal_weight=diagonal_weight)
                lithotype_map[i, j] = decide_new_state(values, lithotype_map[i, j], threshold=effective_threshold,
                                                        neighbour_weights=w, rule=rule, temperature=temperature,
                                                        interaction_matrix=interaction_matrix, rng=rng)

        if nucleation_rate > 0:
            nucleation_mask = rng.random(lithotype_map.shape) < nucleation_rate
            if locked_mask is not None:
                nucleation_mask &= ~locked_mask
            random_states = rng.integers(0, 3, size=lithotype_map.shape)
            lithotype_map[nucleation_mask] = random_states[nucleation_mask]

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



