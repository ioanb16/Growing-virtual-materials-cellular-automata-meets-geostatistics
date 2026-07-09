import numpy as np
import gstools as gs
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def make_gaussian_fields(
    grid_size=100,
    var_1=1, len_scale_1=[30, 30], angles_1=[0, 0],          seed_1=0,
    var_2=1, len_scale_2=[20, 20], angles_2=[np.pi/4, np.pi/4], seed_2=1
):
    """
    Generates two independent Gaussian random fields.
    
    Parameters:
    - grid_size   : size of the grid (default 100)
    - var_1, var_2: variance of the fields (default 1). Does not affect 
                    lithotype proportions due to empirical quantile usage.
    - len_scale_1, len_scale_2: correlation lengths.
    - angles_1, angles_2      : orientations in radians.
    - seed_1, seed_2          : random seeds for reproducibility.
    
    Returns:
    - (field_1, field_2): 2D numpy arrays of shape (grid_size, grid_size)
    """
    x = y = range(grid_size)

    model_1 = gs.Gaussian(dim=2, var=var_1, len_scale=len_scale_1, angles=angles_1)
    model_2 = gs.Gaussian(dim=2, var=var_2, len_scale=len_scale_2, angles=angles_2)

    srf_1 = gs.SRF(model_1)
    srf_2 = gs.SRF(model_2)

    srf_1((x, y), mesh_type='structured', seed=seed_1)
    srf_2((x, y), mesh_type='structured', seed=seed_2)

    return srf_1.field, srf_2.field


def make_lithotype_map(field_1, field_2, Mat1=0.20, Mat2=0.50, Mat3=0.30):
    """
    Truncates two Gaussian fields into a 3-phase lithotype map based on target proportions.
    Proportions must sum to 1.0. Thresholds use conditional empirical quantiles.
    """
    f1 = field_1.ravel()
    f2 = field_2.ravel()

    t1 = np.quantile(f1, Mat1)
    t2 = np.quantile(f2[f1 >= t1], Mat2 / (Mat2 + Mat3))

    lithotype_map = np.zeros(f1.shape[0], dtype=int)
    lithotype_map[(f1 >= t1) & (f2 < t2)] = 1
    lithotype_map[(f1 >= t1) & (f2 >= t2)] = 2

    return lithotype_map.reshape(field_1.shape)


def plot_fields(field_1, field_2, cmap='viridis', figsize=(10, 4)):
    """Plots the two underlying Gaussian fields side-by-side."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    axes[0].imshow(field_1, cmap=cmap)
    axes[0].set_title('Field 1')
    axes[1].imshow(field_2, cmap=cmap)
    axes[1].set_title('Field 2')
    plt.show()


def plot_lithotype_map(lithotype_map, cmap='copper', labels=['Mat1', 'Mat2', 'Mat3'], figsize=(6, 6)):
    """Plots the discrete 3-phase classified lithotype map."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(lithotype_map, cmap=cmap, vmin=0, vmax=2)

    colormap = plt.colormaps[cmap]
    patches = [
        Patch(color=colormap(0/2), label=labels[0]),
        Patch(color=colormap(1/2), label=labels[1]),
        Patch(color=colormap(2/2), label=labels[2])
    ]
    ax.legend(handles=patches, loc='upper right')
    plt.show()