import torch
import numpy as np
import os
import glob
import random
import matplotlib
import imageio
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from qm9 import analyze
##############
### Files ####
###########-->


bond1_radius = {'H': 31, 'C': 76, 'N': 71, 'O': 66, 'F': 57} # covalnt bond in pm for each type of atom https://en.wikipedia.org/wiki/Covalent_radius
bond1_stdv = {'H': 5, 'C': 2, 'N': 2, 'O': 2, 'F': 3}

bond2_radius = {'H': -1000, 'C': 67, 'N': 60, 'O': 57, 'F': 59}
bond3_radius = {'H': -1000, 'C': 60, 'N': 54, 'O': 53, 'F': 53} # Not sure why oxygen has triple bond





def save_xyz_file(
        path, one_hot, charges, positions, id_from=0, name='molecule'):
    try:
        os.makedirs(path)
    except OSError:
        pass
    for batch_i in range(one_hot.size(0)):
        f = open(path + name + '_' + "%03d.txt" % (batch_i + id_from), "w")
        f.write("%d\n\n" % one_hot.size(1))
        atoms = torch.argmax(one_hot[batch_i], dim=1)
        for atom_i in range(one_hot.size(1)):
            atom = atoms[atom_i]
            atom = analyze.atom_decoder[atom]
            f.write("%s %.9f %.9f %.9f\n" % (atom, positions[batch_i, atom_i, 0], positions[batch_i, atom_i, 1], positions[batch_i, atom_i, 2]))
        f.close()


def load_molecule_xyz(file):
    with open(file, encoding='utf8') as f:
        n_atoms = int(f.readline())
        one_hot = torch.zeros(n_atoms, 5)
        charges = torch.zeros(n_atoms, 1)
        positions = torch.zeros(n_atoms, 3)
        f.readline()
        atoms = f.readlines()
        for i in range(n_atoms):
            atom = atoms[i].split(' ')
            atom_type = atom[0]
            one_hot[i, analyze.atom_encoder[atom_type]] = 1
            position = torch.Tensor([float(e) for e in atom[1:]])
            positions[i, :] = position
        return positions, one_hot, charges


def load_xyz_files(path, shuffle=True):
    files = glob.glob(path + "/*.txt")
    if shuffle:
        random.shuffle(files)
    return files

#<----########
### Files ####
##############
def draw_sphere(ax, x, y, z, size, color):
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    xs = size * np.outer(np.cos(u), np.sin(v))
    ys = size * np.outer(np.sin(u), np.sin(v))
    zs = size * np.outer(np.ones(np.size(u)), np.cos(v))
    # for i in range(2):
    #    ax.plot_surface(x+random.randint(-5,5), y+random.randint(-5,5), z+random.randint(-5,5),  rstride=4, cstride=4, color='b', linewidth=0, alpha=0.5)

    ax.plot_surface(x + xs, y + ys, z + zs, rstride=2, cstride=2, color=color, linewidth=0,
                    alpha=1.)
    # # calculate vectors for "vertical" circle
    # a = np.array([-np.sin(elev / 180 * np.pi), 0, np.cos(elev / 180 * np.pi)])
    # b = np.array([0, 1, 0])
    # b = b * np.cos(rot) + np.cross(a, b) * np.sin(rot) + a * np.dot(a, b) * (
    #             1 - np.cos(rot))
    # ax.plot(np.sin(u), np.cos(u), 0, color='k', linestyle='dashed')
    # horiz_front = np.linspace(0, np.pi, 100)
    # ax.plot(np.sin(horiz_front), np.cos(horiz_front), 0, color='k')
    # vert_front = np.linspace(np.pi / 2, 3 * np.pi / 2, 100)
    # ax.plot(a[0] * np.sin(u) + b[0] * np.cos(u), b[1] * np.cos(u),
    #         a[2] * np.sin(u) + b[2] * np.cos(u), color='k', linestyle='dashed')
    # ax.plot(a[0] * np.sin(vert_front) + b[0] * np.cos(vert_front),
    #         b[1] * np.cos(vert_front),
    #         a[2] * np.sin(vert_front) + b[2] * np.cos(vert_front), color='k')
    #
    # ax.view_init(elev=elev, azim=0)


def plot_data3d(positions, atom_type, camera_elev=0, camera_azim=0, save_path=None, spheres_3d=False, bg='black'):

    black = (0, 0, 0)
    white = (1, 1, 1)

    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_aspect('auto')
    ax.view_init(elev=camera_elev, azim=camera_azim)
    if bg == 'black':
        ax.set_facecolor(black)
    else:
        ax.set_facecolor(white)
    #ax.xaxis.pane.set_edgecolor('#D0D0D0')
    ax.xaxis.pane.set_alpha(0)
    ax.yaxis.pane.set_alpha(0)
    ax.zaxis.pane.set_alpha(0)
    ax._axis3don = False

    # draw_sphere(ax, 0, 0, 0, 1)
    # draw_sphere(ax, 1, 1, 1, 1)

    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]
    # Hydrogen, Carbon, Nitrogen, Oxygen, Flourine
    if bg == 'black':
        ax.w_xaxis.line.set_color("black")
    else:
        ax.w_xaxis.line.set_color("white")
    #ax.set_facecolor((1.0, 0.47, 0.42))
    colors_dic = np.array(['#FFFFFF99', 'C7', 'C0', 'C3', 'C1'])
    radius_dic = np.array([0.46, 0.77, 0.77, 0.77, 0.77])
    area_dic = 1500 * radius_dic ** 2
    #areas_dic = sizes_dic * sizes_dic * 3.1416

    areas = area_dic[atom_type]
    radii = radius_dic[atom_type]
    colors = colors_dic[atom_type]

    if spheres_3d:
        for i, j, k, s, c in zip(x, y, z, radii, colors):
            draw_sphere(ax, i.item(), j.item(), k.item(), 0.7 * s, c)
    else:
        ax.scatter(x, y, z, s=areas, alpha=0.9, c=colors)#, linewidths=2, edgecolors='#FFFFFF')

    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            p1 = np.array([x[i], y[i], z[i]])
            p2 = np.array([x[j], y[j], z[j]])
            dist = np.sqrt(np.sum((p1 - p2) ** 2))
            atom1, atom2 = analyze.atom_decoder[atom_type[i]], analyze.atom_decoder[atom_type[j]]
            if analyze.get_bond_order(atom1, atom2, dist):
                if bg == 'black':
                    ax.plot([x[i], x[j]], [y[i], y[j]], [z[i], z[j]], linewidth=(3-2)*2 * 2, c='#FFFFFF')
                else:
                    ax.plot([x[i], x[j]], [y[i], y[j]], [z[i], z[j]],
                            linewidth=(3 - 2) * 2 * 2, c='#666666')
    #plt.show()

    # max_value = positions.abs().max().item()

    axis_lim = 3.2
    ax.set_xlim(-axis_lim, axis_lim)
    ax.set_ylim(-axis_lim, axis_lim)
    ax.set_zlim(-axis_lim, axis_lim)

    dpi = 100 if spheres_3d else 50

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.0, dpi=dpi)

        if spheres_3d:
            img = imageio.imread(save_path)
            img_brighter = np.clip(img * 1.4, 0, 255).astype('uint8')
            imageio.imsave(save_path, img_brighter)
    else:
        plt.show()
    plt.close()


def plot_grid():
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import ImageGrid

    im1 = np.arange(100).reshape((10, 10))
    im2 = im1.T
    im3 = np.flipud(im1)
    im4 = np.fliplr(im2)

    fig = plt.figure(figsize=(10., 10.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(6, 6),  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     )

    for ax, im in zip(grid, [im1, im2, im3, im4]):
        # Iterating over the grid returns the Axes.

        ax.imshow(im)

    plt.show()


def visualize(path, max_num=25, wandb=None, spheres_3d=False):
    files = load_xyz_files(path)[0:max_num]
    for file in files:
        positions, one_hot, charges = load_molecule_xyz(file)
        atom_type = torch.argmax(one_hot, dim=1).numpy()
        plot_data3d(positions, atom_type, save_path=file[:-4] + '.png',
                    spheres_3d=spheres_3d)

        if wandb is not None:
            path = file[:-4] + '.png'
            # Log image(s)
            im = plt.imread(path)
            wandb.log({path: [wandb.Image(im, caption=path)]})


def visualize_chain(path, wandb=None, spheres_3d=False):
    files = load_xyz_files(path)
    files = sorted(files)
    save_paths = []

    print(f'Visualizing chain using files: {files}')
    for file in files:
        positions, one_hot, charges = load_molecule_xyz(file)
        atom_type = torch.argmax(one_hot, dim=1).numpy()
        fn = file[:-4] + '.png'
        plot_data3d(positions, atom_type, save_path=fn, spheres_3d=spheres_3d)
        save_paths.append(fn)

    imgs = [imageio.imread(fn) for fn in save_paths]
    dirname = os.path.dirname(save_paths[0])
    gif_path = dirname + '/output.gif'
    print(f'Creating gif with {len(imgs)} images')
    # Add the last frame 10 times so that the final result remains temporally.
    # imgs.extend([imgs[-1]] * 10)
    imageio.mimsave(gif_path, imgs, subrectangles=True)

    if wandb is not None:
        wandb.log({gif_path: [wandb.Video(gif_path, caption=gif_path)]})


if __name__ == '__main__':
    #plot_grid()
    import qm9.dataset as dataset
    matplotlib.use('macosx')

    task = "plot_chain"

    if task == "visualize_molecules":
        dataloaders, charge_scale = dataset.retrieve_dataloaders(batch_size=1)
        for i, data in enumerate(dataloaders['train']):
            positions = data['positions'].view(-1, 3)
            positions_centered = positions - positions.mean(dim=0, keepdim=True)
            one_hot = data['one_hot'].view(-1, 5).type(torch.float32)
            atom_type = torch.argmax(one_hot, dim=1).numpy()

            plot_data3d(positions_centered, atom_type, spheres_3d=True)
    elif task == "plot_chain":
        visualize_chain(path="../outputs/here_we_go2_resume_best_batch/eval/chain", spheres_3d=False)
    else:
        raise Exception("Wrong task")
