import numpy as np
import gstools as gs
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def make_gaussian_fields(
    grid_size=100,
    var_1=1, len_scale_1=[30, 30], angles_1=[0, 0],          seed_1=0,
    var_2=1, len_scale_2=[20, 20], angles_2=[np.pi/4, np.pi/4], seed_2=1,
    help=False
):
    if help:
        print("""
        make_gaussian_fields() parameters:
        grid_size   : size of the grid (default 100)
        var_1       : variance of field 1 (default 1)
                      note: does not affect lithotype proportions since
                      make_lithotype_map uses empirical quantiles
        len_scale_1 : correlation length of field 1 (default [30, 30])
        angles_1    : orientation of field 1 in radians (default [0, 0])
        seed_1      : random seed for field 1 (default 0)
        var_2       : variance of field 2 (default 1) — same note as var_1
        len_scale_2 : correlation length of field 2 (default [20, 20])
        angles_2    : orientation of field 2 in radians (default [pi/4, pi/4])
        seed_2      : random seed for field 2 (default 1)

        Returns (field_1, field_2) as 2D numpy arrays of shape (grid_size, grid_size)
        """)
        return

    x = y = range(grid_size)

    model_1 = gs.Gaussian(dim=2, var=var_1, len_scale=len_scale_1, angles=angles_1)
    model_2 = gs.Gaussian(dim=2, var=var_2, len_scale=len_scale_2, angles=angles_2)

    srf_1 = gs.SRF(model_1)
    srf_2 = gs.SRF(model_2)

    srf_1((x, y), mesh_type='structured', seed=seed_1)
    srf_2((x, y), mesh_type='structured', seed=seed_2)

    return srf_1.field, srf_2.field


def make_lithotype_map(
    field_1, field_2,
    Mat1=0.20, Mat2=0.50, Mat3=0.30,
    help=False
):
    if help:
        print("""
        make_lithotype_map() parameters:
        field_1 : first Gaussian field (from make_gaussian_fields)
        field_2 : second Gaussian field (from make_gaussian_fields)
        Mat1    : proportion of material 1 (default 0.20)
        Mat2    : proportion of material 2 (default 0.50)
        Mat3    : proportion of material 3 (default 0.30)
        note    : proportions must sum to 1.0
                  thresholds use conditional empirical quantiles so
                  output proportions match targets exactly regardless
                  of field variance or correlation length
        """)
        return

    f1 = field_1.ravel()
    f2 = field_2.ravel()

    t1 = np.quantile(f1, Mat1)
    t2 = np.quantile(f2[f1 >= t1], Mat2 / (Mat2 + Mat3))

    lithotype_map = np.zeros(f1.shape[0], dtype=int)
    lithotype_map[(f1 >= t1) & (f2 < t2)] = 1
    lithotype_map[(f1 >= t1) & (f2 >= t2)] = 2

    return lithotype_map.reshape(field_1.shape)


def plot_fields(
    field_1, field_2,
    cmap='viridis', figsize=(10, 4),
    help=False
):
    if help:
        print("""
        plot_fields() parameters:
        field_1 : first Gaussian field
        field_2 : second Gaussian field
        cmap    : colormap (default 'viridis')
        figsize : figure size (default (10, 4))
        """)
        return

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    axes[0].imshow(field_1, cmap=cmap)
    axes[0].set_title('Field 1')
    axes[1].imshow(field_2, cmap=cmap)
    axes[1].set_title('Field 2')
    plt.show()


def plot_lithotype_map(
    lithotype_map,
    cmap='copper', labels=['Mat1', 'Mat2', 'Mat3'], figsize=(6, 6),
    help=False
):
    if help:
        print("""
        plot_lithotype_map() parameters:
        lithotype_map : classified map (from make_lithotype_map)
        cmap          : colormap (default 'copper')
        labels        : names for phases 0, 1, 2 (default ['Mat1','Mat2','Mat3'])
        figsize       : figure size (default (6, 6))
        """)
        return

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