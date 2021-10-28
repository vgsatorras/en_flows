import numpy as np
import matplotlib.pyplot as plt


def free_energy_bootstrap(D, l, r, n, sample=100, weights=None, bias=None, temperature=1.0):
    """ Bootstrapped free energy calculation

    If D is a single array, bootstraps by sample. If D is a list of arrays, bootstraps by trajectories

    Parameters
    ----------
    D : array of list of arrays
        Samples in the coordinate in which we compute the free energy
    l : float
        leftmost bin boundary
    r : float
        rightmost bin boundary
    n : int
        number of bins
    sample : int
        number of bootstraps
    weights : None or arrays matching D
        sample weights
    bias : function
        if not None, the given bias will be removed.

    Returns
    -------
    bin_means : array((nbins,))
        mean positions of bins
    Es : array((sample, nbins))
        for each bootstrap the free energies of bins.

    """
    bins = np.linspace(l, r, n)
    Es = []
    I = np.arange(len(D))
    by_traj = isinstance(D, list)
    for s in range(sample):
        Isel = np.random.choice(I, size=len(D), replace=True)
        if by_traj:
            Dsample = np.concatenate([D[i] for i in Isel])
            Wsample = None
            if weights is not None:
                Wsample = np.concatenate([weights[i] for i in Isel])
            Psample, _ = np.histogram(Dsample, bins=bins, weights=Wsample, density=True)
        else:
            Dsample = D[Isel]
            Wsample = None
            if weights is not None:
                Wsample = weights[Isel]
            Psample, _ = np.histogram(Dsample, bins=bins, weights=Wsample, density=True)
        Es.append(-np.log(Psample))
    Es = np.vstack(Es)
    Es -= Es.mean(axis=0).min()
    bin_means = 0.5 * (bins[:-1] + bins[1:])

    if bias is not None:
        B = bias(bin_means) / temperature
        Es -= B

    return bin_means, Es  # / temperature


def mean_finite_(x, min_finite=1):
    """ Computes mean over finite values """
    isfin = np.isfinite(x)
    if np.count_nonzero(isfin) > min_finite:
        return np.mean(x[isfin])
    else:
        return np.nan


def std_finite_(x, min_finite=2):
    """ Computes mean over finite values """
    isfin = np.isfinite(x)
    if np.count_nonzero(isfin) > min_finite:
        return np.std(x[isfin])
    else:
        return np.nan


def mean_finite(x, axis=None, min_finite=1):
    if axis is None:
        return mean_finite_(x)
    if axis == 0 or axis == 1:
        M = np.zeros((x.shape[axis - 1],))
        for i in range(x.shape[axis - 1]):
            if axis == 0:
                M[i] = mean_finite_(x[:, i])
            else:
                M[i] = mean_finite_(x[i])
        return M
    else:
        raise NotImplementedError('axis value not implemented:', axis)


def std_finite(x, axis=None, min_finite=2):
    if axis is None:
        return mean_finite_(x)
    if axis == 0 or axis == 1:
        S = np.zeros((x.shape[axis - 1],))
        for i in range(x.shape[axis - 1]):
            if axis == 0:
                S[i] = std_finite_(x[:, i])
            else:
                S[i] = std_finite_(x[i])
        return S
    else:
        raise NotImplementedError('axis value not implemented:', axis)

def distance(x, particles, dims):
    """Returns distances between particles"""
    x = x.reshape(-1, particles, dims)
    distances = np.sqrt(np.power((np.expand_dims(x, 1) - np.expand_dims(x, 2)), 2).sum(-1))
    # take only the relevant terms (upper triangular)
    row_idx, col_idx = np.triu_indices(distances.shape[1], 1)
    return distances[:, row_idx, col_idx]

def energy_plot(x, potential):
    if potential == "dimer":
        d = x - 3
        d2 = d ** 2
        d4 = d2 ** 2
        return -4. * d2 + 1. * d4

    elif potential == "prince":
        """Prince energy"""
        x = x - 2
        x1 = x ** 8
        x2 = np.exp(-80. * x * x)
        x3 = np.exp(-80. * (x - 0.5) ** 2)
        x4 = np.exp(-40. * (x + 0.5) ** 2)
        return 4 * (1. * x1 + 0.8 * x2 + 0.2 * x3 + 0.5 * x4)

def plot_system(x, n_particles, n_dimensions, n_plots=(4, 4),
                lim=4):
    """Plot system of 4 particles."""
    x = x.reshape(-1, n_particles, n_dimensions)
    # np.random.shuffle(x)
    if n_dimensions > 2:
        raise NotImplementedError("dimension must be <= 2")
    plt.figure(figsize=(n_plots[1] * 2, n_plots[0] * 2))
    for i in range(np.prod(n_plots)):
        plt.subplot(*n_plots, i+1)
        plt.scatter(*x[i].T)
        #plt.xticks([])
        #plt.yticks([])
        plt.tight_layout()
        plt.xlim((-lim, lim))
        plt.ylim((-lim, lim))

############################## Plotting #######################################


def create_plots_2part(file, filename, fixed_z, x_samples, delta_logp, potential, epoch):
    """Plots for 2 particle dimer."""
    data_shape = x_samples.shape[1]
    n_dims = data_shape // 2
    fig_filename = os.path.join(file, filename, "pos" + "{:04d}.jpg".format(epoch))
    utils.makedirs(os.path.dirname(fig_filename))

    generated_samples = x_samples.view(-1, data_shape).data.cpu().numpy()

    # plot dimer
    if n_dims == 2:
        plt.figure(figsize=(5, 4))
        for point in generated_samples[100:200]:
            plt.plot([point[0], point[2]], [point[1], point[3]], 'k-', lw=0.5)
            plt.scatter(point[0], point[1], c='r', marker='x')
            plt.scatter(point[2], point[3], c='b', marker='x')
        plt.savefig(fig_filename, bbox_inches='tight')

    generated_samples = distance(x_samples.data.cpu().numpy(), 2, n_dims).reshape(-1)
    jac = - delta_logp.data.cpu().view(-1).numpy()
    plt.figure(figsize=(5, 4))
    h, b = np.histogram(generated_samples, bins=100)
    Eh = -np.log(h)
    Eh = Eh - Eh.min()
    if potential == "prince":
        d = np.linspace(0, 4, num=200)
    elif potential == "dimer":
        d = np.linspace(0, 6, num=200)

    energies = energy_plot(d, potential) - (n_dims - 1) * np.log(d)

    plt.plot(d, energies - energies.min(), linewidth=2, )
    bin_means = 0.5 * (b[:-1] + b[1:])
    plt.plot(bin_means, Eh)
    plt.ylim(0, 6)
    fig_filename = os.path.join(file, filename, "{:04d}.jpg".format(epoch))
    plt.savefig(fig_filename, bbox_inches='tight')

    # reweighting factor
    print(jac.shape, energy_plot(generated_samples, potential).shape, (0.5 * np.sum(fixed_z.view(-1, data_shape).data.cpu().numpy() ** 2,  axis=1)).shape)
    log_w = - energy_plot(generated_samples, potential) + 0.5 * np.sum(fixed_z.view(-1, data_shape).data.cpu().numpy() ** 2, axis=1) + jac
    print(np.mean(energy_plot(generated_samples, potential)), np.mean(0.5 * np.sum(fixed_z.view(-1, data_shape).data.cpu().numpy() ** 2, axis=1)), np.mean(jac))
    print(min(log_w), max(log_w))

    plt.figure(figsize=(5, 4))
    if potential == "prince":
        bin_means, Es = free_energy_bootstrap(generated_samples, -4, 4, 100, sample=100, weights=np.exp(log_w))
    elif potential == "dimer":
        bin_means, Es = free_energy_bootstrap(generated_samples, 0, 10, 100, sample=100,
                                              weights=np.exp(log_w))

    plt.plot(d, energies - energies.min(), linewidth=2, )
    Emean = mean_finite(Es, axis=0)
    Estd = std_finite(Es, axis=0)
    plt.fill_between(bin_means, Emean, Emean + Estd, alpha=0.2)
    plt.fill_between(bin_means, Emean, Emean - Estd, alpha=0.2)
    plt.plot(bin_means, Emean)
    plt.ylim(0, 6)
    fig_filename = os.path.join(file, filename, "rew" + "{:04d}.jpg".format(epoch))
    utils.makedirs(os.path.dirname(fig_filename))
    plt.savefig(fig_filename, bbox_inches='tight')


def create_plots_3part(file, filename, fixed_z, x_samples, delta_logp, potential, epoch):
    fig_filename = os.path.join(file, filename, "pos" + "{:04d}.jpg".format(epoch))
    utils.makedirs(os.path.dirname(fig_filename))
    data_shape = x_samples.shape[1]
    n_dims = data_shape // 3
    generated_samples = x_samples.view(-1, data_shape).data.cpu().numpy()

    # plot mean of dimer
    plt.figure(figsize=(5, 4))
    for point in generated_samples[1000:1020]:
        plt.plot([point[0], point[2]], [point[1], point[3]], 'k-', lw=0.2)
        plt.plot([point[0], point[4]], [point[1], point[5]], 'k-', lw=0.2)
        plt.plot([point[2], point[4]], [point[3], point[5]], 'k-', lw=0.2)
        plt.scatter(point[0], point[1], c='r', marker='x')
        plt.scatter(point[2], point[3], c='b', marker='x')
        plt.scatter(point[4], point[5], c='g', marker='x')

    plt.savefig(fig_filename, bbox_inches='tight')



    plt.figure(figsize=(5, 4))
    dists = distance(generated_samples, 3, n_dims)
    plt.hist(np.sum(energy_plot(dists, potential), axis=-1), bins=1000, density=True)
    plt.xlabel('Energy')
    plt.ylabel('Probability')
    plt.xlim(-13, 20)

    fig_filename = os.path.join(file, filename, "energy" + "{:04d}.jpg".format(epoch))
    plt.savefig(fig_filename, bbox_inches='tight')

    dists = distance(generated_samples, 3, n_dims)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xs = dists[:, 0][np.where(dists[:, 2] < 3)]
    ys = dists[:, 1][np.where(dists[:, 2] < 3)]
    zs = dists[:, 2][np.where(dists[:, 2] < 3)]
    ax.scatter(xs, ys, zs, c="r", marker='.', s=[1, 1, 1])

    ax.set_xlabel('Dist 1')
    ax.set_ylabel('Dist 2')
    ax.set_zlabel('Dist 3')
    # ax.set_zlim(0,3)
    fig_filename = os.path.join(file, filename, "bottom" + "{:04d}.jpg".format(epoch))
    utils.makedirs(os.path.dirname(fig_filename))
    plt.savefig(fig_filename, bbox_inches='tight')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xs = dists[:, 0][np.where(dists[:, 2] > 3)]
    ys = dists[:, 1][np.where(dists[:, 2] > 3)]
    zs = dists[:, 2][np.where(dists[:, 2] > 3)]
    ax.scatter(xs, ys, zs, c="r", marker='.', s=[1, 1, 1])

    ax.set_xlabel('Dist 1')
    ax.set_ylabel('Dist 2')
    ax.set_zlabel('Dist 3')
    # ax.set_zlim(0,3)
    fig_filename = os.path.join(file, filename, "top" + "{:04d}.jpg".format(epoch))
    utils.makedirs(os.path.dirname(fig_filename))
    plt.savefig(fig_filename, bbox_inches='tight')

    # TODO reweighting
    # projection on axis
    for i, a in enumerate("xyz"):
        plt.figure(figsize=(5, 4))
        h, b = np.histogram(dists[:, i], bins=100)
        Eh = -np.log(h)
        Eh = Eh - Eh.min()
        d = np.linspace(0, 6, num=200)
        energies = energy_plot(d, potential) - np.log(d)
        plt.plot(d, energies - energies.min(), linewidth=2, )
        bin_means = 0.5 * (b[:-1] + b[1:])
        plt.plot(bin_means, Eh)
        plt.ylim(0, 6)

        fig_filename = os.path.join(file, filename, a+"{:04d}.jpg".format(epoch))
        utils.makedirs(os.path.dirname(fig_filename))
        plt.savefig(fig_filename, bbox_inches='tight')


def create_plots_4part(file, filename, fixed_z, x_samples, delta_logp, potential, epoch):
    fig_filename = os.path.join(file, filename, "pos" + "{:04d}.jpg".format(epoch))
    utils.makedirs(os.path.dirname(fig_filename))
    data_shape = x_samples.shape[1]
    n_dims = data_shape // 4
    generated_samples = x_samples.view(-1, data_shape).data.cpu().numpy()


    plot_system(generated_samples, 4, n_dims)

    plt.savefig(fig_filename, bbox_inches='tight')



    plt.figure(figsize=(5, 4))
    dists = distance(generated_samples, 4, n_dims)
    plt.hist(np.sum(energy_plot(dists, potential), axis=-1), bins=1000, density=True)
    plt.xlabel('Energy')
    plt.ylabel('Probability')
    plt.xlim(-25, 20)

    fig_filename = os.path.join(file, filename, "energy" + "{:04d}.jpg".format(epoch))
    plt.savefig(fig_filename, bbox_inches='tight')

    n_particles = 4
    n_dimensions = 2
    plt.figure(figsize=(10, 10))
    # dists = all_distances(torch.Tensor(data).view(-1, n_particles, n_dimensions)).numpy()
    plot_idx = 1
    for i in range(n_particles):
        for j in range(n_particles - 1):
            values, bins = np.histogram(dists[:, i + j], bins=100)
            values = -np.log(values)
            plt.subplot(n_particles, n_particles - 1, plot_idx)
            d = np.linspace(0, 6, num=200)
            energies = energy_plot(d, potential) - np.log(d)
            plt.plot(d, energies - energies.min(), linewidth=2, )
            plt.ylim(0, 6)
            plt.plot((bins[1:] + bins[:-1]) / 2, values - values.min())

            plot_idx += 1
    # ax.set_zlim(0,3)
    fig_filename = os.path.join(file, filename, "Distances" + "{:04d}.jpg".format(epoch))
    utils.makedirs(os.path.dirname(fig_filename))
    plt.savefig(fig_filename, bbox_inches='tight')

