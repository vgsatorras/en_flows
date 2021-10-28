import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation
from IPython.display import display, clear_output

from .distances import distance_vectors, distances_from_vectors


def plot_system(
    x, n_particles, n_dimensions, n_plots=(4, 4), lim=5, randomized=True,
    fig=None, axes=None, potential=None
):
    if isinstance(x, torch.Tensor):
        x = x.detach().numpy()
    if randomized:
        idxs = np.arange(x.shape[0])
        np.random.shuffle(idxs)
        x = x[idxs]
    x = x.reshape(-1, n_particles, n_dimensions)
    if n_dimensions > 2:
        raise NotImplementedError("dimension must be <= 2")
    
    if fig is None or axes is None:
        fig, axes = plt.subplots(*n_plots, figsize=(n_plots[1] * 2, n_plots[0] * 2))

    k=0
    for i in range(n_plots[0]):
        for j in range(n_plots[1]):
#             axes[i][j].subplot(*n_plots, i + 1)
            axes[i][j].scatter(*x[k].T)
            axes[i][j].set_xticks([])
            axes[i][j].set_yticks([])
#             axes[i][j].tight_layout()
            axes[i][j].set_xlim((-lim, lim))
            axes[i][j].set_ylim((-lim, lim))
            if potential is not None:
                energy = potential.energy([x[k]])[0]
                energy = np.around(energy, 2)
                axes[i][j].set_title("{:.2f}".format(energy))
            k+=1
    return fig, axes

def plot_system_results(
        x, x_bg, n_particles, n_dimensions, n_plots=(4, 4), lim=5, randomized=True,
        fig=None, axes=None, potential=None, name=""
):
    """
    Plot particle positions of BG and corresponding minima positions
    :param x: concatenated array of positions of

    """
    if isinstance(x, torch.Tensor):
        x = x.detach().numpy()
    if randomized:
        idxs = np.arange(x.shape[0])
        np.random.shuffle(idxs)
        x = x[idxs]
    x = x.reshape(-1, n_particles, n_dimensions)
    x_bg = x_bg.reshape(-1, n_particles, n_dimensions)
    if n_dimensions > 2:
        raise NotImplementedError("dimension must be <= 2")

    if fig is None or axes is None:
        fig, axes = plt.subplots(*n_plots, figsize=(n_plots[1] * 2, n_plots[0] * 2))

    k=0
    matplotlib.pyplot.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.4)
    for i in range(n_plots[0]):
        for j in range(n_plots[1]):
            #             axes[i][j].subplot(*n_plots, i + 1)
            if i == 0:
                axes[i][j].scatter(*x[j].T, c='b', s=100)
            else:
                axes[i][j].scatter(*x_bg[j].T, c='r', s=100)
            axes[i][j].set_xticks([])
            axes[i][j].set_yticks([])
            #             axes[i][j].tight_layout()
            axes[i][j].set_xlim((-lim, lim))
            axes[i][j].set_ylim((-lim, lim))
            if potential is not None:
                if i == 0:
                    energy = potential.energy([x[j]])[0]
                else:
                    energy = potential.energy([x_bg[j]])[0]
                # every distance is counted twice
                energy = np.around(energy, 2) / 2
                axes[i][j].set_title("{:.2f}".format(energy), fontsize=22)
            k+=1
    return fig, axes


def animate_system():
    
    if isinstance(x, torch.Tensor):
        x = x.detach().numpy()
    if randomized:
        idxs = np.arange(x.shape[0])
        np.random.shuffle(idxs)
        x = x[idxs]
    x = x.reshape(-1, n_particles, n_dimensions)
    if n_dimensions > 2:
        raise NotImplementedError("dimension must be <= 2")
    fig = plt.figure(figsize=(n_plots[1] * 2, n_plots[0] * 2))
    for i in range(np.prod(n_plots)):
        fig.subplot(*n_plots, i + 1)
        fig.scatter(*x[i].T)
        fig.xticks([])
        fig.yticks([])
        fig.tight_layout()
        fig.xlim((-lim, lim))
        fig.ylim((-lim, lim))
        
    
    t = np.linspace(0,2*np.pi)
    x = np.sin(t)

    fig, ax = plt.subplots()
    l, = ax.plot([0,2*np.pi],[-1,1])

    animate = lambda i: l.set_data(t[:i], x[:i])

    for i in range(len(x)):
        animate(i)
        clear_output(wait=True)
        display(fig)

    plt.show()


def plot_distance_hist(x, n_particles, n_dimensions, bins=100, xs=None, ys=None):
    if not isinstance(x, torch.Tensor):
        x = torch.Tensor(x)
    x = x.contiguous()
    plt.figure(figsize=(10, 10))
    dists = distances_from_vectors(
        distance_vectors(x.view(-1, n_particles, n_dimensions))
    ).cpu().detach().numpy()
    plot_idx = 1
    for i in range(n_particles):
        for j in range(n_particles - 1):
            plt.subplot(n_particles, n_particles - 1, plot_idx)
            plt.hist(dists[:, i, j], bins=bins, density=True)
            if xs is not None and ys     is not None:
                plt.plot(xs, np.exp(-ys) / xs**(n_dimensions - 1) / 32.6)
            plt.yticks([], [])
            plot_idx += 1
