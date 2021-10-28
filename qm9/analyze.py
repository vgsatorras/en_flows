import qm9.dataset as dataset
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sp_stats


atom_decoder = ['H', 'C', 'N', 'O', 'F']
atom_encoder = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}

#Bond lengths from http://www.wiredchemist.com/chemistry/data/bond_energies_lengths.html
bonds1 = {'H': {'H': 74, 'C': 109, 'N': 101, 'O': 96, 'F': 92},
          'C': {'H': 109, 'C': 154 , 'N': 147, 'O': 143, 'F': 135},
          'N': {'H': 101, 'C': 147, 'N': 145, 'O': 140, 'F': 136},
          'O': {'H': 96, 'C': 143, 'N': 140, 'O': 148, 'F': 142},
          'F': {'H': 92, 'C': 135, 'N': 136, 'O': 142, 'F': 142}}

bonds2 = {'H': {'H': -1000, 'C': -1000, 'N': -1000, 'O': -1000, 'F': -1000},
          'C': {'H': -1000, 'C': 134, 'N': 129, 'O': 120, 'F': -1000},
          'N': {'H': -1000, 'C': 129, 'N': 125, 'O': 121, 'F': -1000},
          'O': {'H': -1000, 'C': 120, 'N': 121, 'O': 121, 'F': -1000},
          'F': {'H': -1000, 'C': -1000, 'N': -1000, 'O': -1000, 'F': -1000}}

bonds3 = {'H': {'H': -1000, 'C': -1000, 'N': -1000, 'O': -1000, 'F': -1000},
          'C': {'H': -1000, 'C': 120, 'N': 116, 'O': 113, 'F': -1000},
          'N': {'H': -1000, 'C': 116, 'N': 110, 'O': -1000, 'F': -1000},
          'O': {'H': -1000, 'C': 113, 'N': -1000, 'O': -1000, 'F': -1000},
          'F': {'H': -1000, 'C': -1000, 'N': -1000, 'O': -1000, 'F': -1000}}
stdv = {'H': 5, 'C': 1, 'N': 1, 'O': 2, 'F': 3}
margin1, margin2, margin3 = 10, 5, 3

allowed_bonds = {'H': 1, 'C': 4, 'N': 3, 'O': 2, 'F': 1}

#############
## Histograms

analyzed = {'n_nodes': {22: 3393, 17: 13025, 23: 4848, 21: 9970, 19: 13832, 20: 9482, 16: 10644, 13: 3060, 15: 7796, 25: 1506, 18: 13364, 12: 1689, 11: 807, 24: 539, 14: 5136, 26: 48, 7: 16, 10: 362, 8: 49, 9: 124, 27: 266, 4: 4, 29: 25, 6: 9, 5: 5, 3: 1},
            'atom_types': {1: 635559, 2: 101476, 0: 923537, 3: 140202, 4: 2323},
            'distances': [903054, 307308, 111994, 57474, 40384, 29170, 47152, 414344, 2202212, 573726, 1490786, 2970978, 756818, 969276, 489242, 1265402, 4587994, 3187130, 2454868, 2647422, 2098884, 2001974, 1625206, 1754172, 1620830, 1710042, 2133746, 1852492, 1415318, 1421064, 1223156, 1322256, 1380656, 1239244, 1084358, 981076, 896904, 762008, 659298, 604676, 523580, 437464, 413974, 352372, 291886, 271948, 231328, 188484, 160026, 136322, 117850, 103546, 87192, 76562, 61840, 49666, 43100, 33876, 26686, 22402, 18358, 15518, 13600, 12128, 9480, 7458, 5088, 4726, 3696, 3362, 3396, 2484, 1988, 1490, 984, 734, 600, 456, 482, 378, 362, 168, 124, 94, 88, 52, 44, 40, 18, 16, 8, 6, 2, 0, 0, 0, 0, 0, 0, 0]
}

analyzed_19 ={'atom_types': {1: 93818, 3: 21212, 0: 139496, 2: 8251, 4: 26},
            'distances': [0, 0, 0, 0, 0, 0, 0, 22566, 258690, 16534, 50256, 181302, 19676, 122590, 23874, 54834, 309290, 205426, 172004, 229940, 193180, 193058, 161294, 178292, 152184, 157242, 189186, 150298, 125750, 147020, 127574, 133654, 142696, 125906, 98168, 95340, 88632, 80694, 71750, 64466, 55740, 44570, 42850, 36084, 29310, 27268, 23696, 20254, 17112, 14130, 12220, 10660, 9112, 7640, 6378, 5350, 4384, 3650, 2840, 2362, 2050, 1662, 1414, 1216, 966, 856, 492, 516, 420, 326, 388, 326, 236, 140, 130, 92, 62, 52, 78, 56, 24, 8, 10, 12, 18, 2, 10, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
}

class Histogram_discrete:
    def __init__(self, name='histogram'):
        self.name = name
        self.bins = {}

    def add(self, elements):
        for e in elements:
            if e in self.bins:
                self.bins[e] += 1
            else:
                self.bins[e] = 1

    def normalize(self):
        total = 0.
        for key in self.bins:
            total += self.bins[key]
        for key in self.bins:
            self.bins[key] = self.bins[key] / total

    def plot(self, save_path=None):
        width = 1  # the width of the bars
        fig, ax = plt.subplots()
        x, y = [], []
        for key in self.bins:
            x.append(key)
            y.append(self.bins[key])

        ax.bar(x, y, width)
        plt.title(self.name)
        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()


class Histogram_cont:
    def __init__(self, num_bins=100, range=[0., 13.], name='histogram', ignore_zeros=False):
        self.name = name
        self.bins = [0] * num_bins
        self.range = range
        self.ignore_zeros = ignore_zeros

    def add(self, elements):
        for e in elements:
            if not self.ignore_zeros or e > 1e-8:
                i = int(float(e) / self.range[1] * len(self.bins))
                i = min(i, len(self.bins) - 1)
                self.bins[i] += 1

    def plot(self, save_path=None):
        width = (self.range[1] - self.range[0])/len(self.bins) # the width of the bars
        fig, ax = plt.subplots()

        x = np.linspace(self.range[0], self.range[1], num=len(self.bins) + 1)[:-1] + width / 2
        ax.bar(x, self.bins, width)
        plt.title(self.name)

        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()


    def plot_both(self, hist_b, save_path=None, wandb=None):
        ## TO DO: Check if the relation of bins and linspace is correct
        hist_a = normalize_histogram(self.bins)
        hist_b = normalize_histogram(hist_b)

        #width = (self.range[1] - self.range[0]) / len(self.bins)  # the width of the bars
        fig, ax = plt.subplots()
        x = np.linspace(self.range[0], self.range[1], num=len(self.bins) + 1)[:-1]
        ax.step(x, hist_b)
        ax.step(x, hist_a)
        ax.legend(['True', 'Learned'])
        plt.title(self.name)

        if save_path is not None:
            plt.savefig(save_path)
            if wandb is not None:
                if wandb is not None:
                    # Log image(s)
                    im = plt.imread(save_path)
                    wandb.log({save_path: [wandb.Image(im, caption=save_path)]})
        else:
            plt.show()
        plt.close()


def normalize_histogram(hist):
    hist = np.array(hist)
    prob = hist / np.sum(hist)
    return prob


def coord2distances(x):
    x = x.unsqueeze(2)
    x_t = x.transpose(1, 2)
    dist = (x - x_t) ** 2
    dist = torch.sqrt(torch.sum(dist, 3))
    dist = dist.flatten()
    return dist


def earth_mover_distance(h1, h2):
    p1 = normalize_histogram(h1)
    p2 = normalize_histogram(h2)

    distance = sp_stats.wasserstein_distance(p1, p2)
    return distance


def kl_divergence(p1, p2):
    return np.sum(p1*np.log(p1 / p2))


def kl_divergence_sym(h1, h2):
    p1 = normalize_histogram(h1) + 1e-10
    p2 = normalize_histogram(h2) + 1e-10

    kl = kl_divergence(p1, p2)
    kl_flipped = kl_divergence(p2, p1)

    return (kl + kl_flipped) / 2.


def js_divergence(h1, h2):
    p1 = normalize_histogram(h1) + 1e-10
    p2 = normalize_histogram(h2) + 1e-10

    M = (p1 + p2)/2
    js = (kl_divergence(p1, M) + kl_divergence(p2, M)) / 2
    return js


def main_analyze_qm9(n_atoms=None):
    batch_size = 128
    dataloaders, charge_scale = dataset.retrieve_dataloaders(batch_size=batch_size, filter_n_atoms=n_atoms)

    hist_nodes = Histogram_discrete('Histogram # nodes')
    hist_atom_type = Histogram_discrete('Histogram of atom types')
    hist_dist = Histogram_cont(name='Histogram relative distances', ignore_zeros=True)

    for i, data in enumerate(dataloaders['train']):
        print(i * batch_size)

        # Histogram num_nodes
        num_nodes = torch.sum(data['atom_mask'], dim=1)
        num_nodes = list(num_nodes.numpy())
        hist_nodes.add(num_nodes)

        #Histogram edge distances
        x = data['positions']*data['atom_mask'].unsqueeze(2)
        dist = coord2distances(x)
        hist_dist.add(list(dist.numpy()))

        # Histogram of atom types
        one_hot = data['one_hot'].double()
        atom = torch.argmax(one_hot, 2)
        atom = atom.flatten()
        mask = data['atom_mask'].flatten()
        masked_atoms = list(atom[mask].numpy())
        hist_atom_type.add(masked_atoms)

    hist_dist.plot()
    hist_dist.plot_both(hist_dist.bins[::-1])
    print("KL divergence A %.4f" % kl_divergence_sym(hist_dist.bins, hist_dist.bins[::-1]))
    print("KL divergence B %.4f" % kl_divergence_sym(hist_dist.bins, hist_dist.bins))
    print(hist_dist.bins)
    hist_nodes.plot()
    print(hist_nodes.bins)
    hist_atom_type.plot()
    print(hist_atom_type.bins)

## Hystograms
#############


############################
# Validity and bond analysis

def get_bond_order(atom1, atom2, distance):
    distance = 100 * distance  # We change the metric

    # margin1, margin2 and margin3 have been tuned to maximize the stability of the QM9 true samples
    if distance < bonds1[atom1][atom2] + margin1:
        thr_bond2 = bonds2[atom1][atom2] + margin2
        if distance < thr_bond2:
            thr_bond3 = bonds3[atom1][atom2] + margin3
            if distance < thr_bond3:
                return 3
            return 2
        return 1
    return 0


def check_stability(positions, atom_type, debug=False):
    assert len(positions.shape) == 2
    assert positions.shape[1] == 3

    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]

    nr_bonds = np.zeros(len(x), dtype='int')

    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            p1 = np.array([x[i], y[i], z[i]])
            p2 = np.array([x[j], y[j], z[j]])
            dist = np.sqrt(np.sum((p1 - p2) ** 2))
            atom1, atom2 = atom_decoder[atom_type[i]], atom_decoder[
                atom_type[j]]
            order = get_bond_order(atom1, atom2, dist)
            nr_bonds[i] += order
            nr_bonds[j] += order

    nr_stable_bonds = 0
    for atom_type_i, nr_bonds_i in zip(atom_type, nr_bonds):
        is_stable = allowed_bonds[atom_decoder[atom_type_i]] == nr_bonds_i
        if is_stable == False and debug:
            print("Invalid bonds for molecule %s with %d bonds" % (atom_decoder[atom_type_i], nr_bonds_i))
        nr_stable_bonds += int(is_stable)

    molecule_stable = nr_stable_bonds == len(x)
    return molecule_stable, nr_stable_bonds, len(x)


def main_check_stability():
    import qm9.dataset as dataset
    batch_size = 32
    dataloaders, charge_scale = dataset.retrieve_dataloaders(batch_size=batch_size)

    def test_validity_for(dataloader):
        count_mol_stable = 0
        count_atm_stable = 0
        count_mol_total = 0
        count_atm_total = 0
        for data in dataloader:
            for i in range(data['positions'].size(0)):
                positions = data['positions'][i].view(-1, 3)
                one_hot = data['one_hot'][i].view(-1, 5).type(torch.float32)
                mask = data['atom_mask'][i].flatten()
                positions, one_hot = positions[mask], one_hot[mask]
                atom_type = torch.argmax(one_hot, dim=1).numpy()
                is_stable, nr_stable, total = check_stability(positions, atom_type)

                count_atm_stable += nr_stable
                count_atm_total += total

                count_mol_stable += int(is_stable)
                count_mol_total += 1

            print(f"Stable molecules "
                  f"{100. * count_mol_stable/count_mol_total:.2f} \t"
                  f"Stable atoms: "
                  f"{100. * count_atm_stable/count_atm_total:.2f} \t"
                  f"Counted molecules {count_mol_total}/{len(dataloader)*batch_size}")

    print('For train')
    test_validity_for(dataloaders['train'])

    print('For test')
    test_validity_for(dataloaders['test'])


def analyze_stability_for_molecules(molecule_list):
    n_samples = len(molecule_list)
    molecule_stable_list = []

    molecule_stable = 0
    nr_stable_bonds = 0
    n_atoms = 0

    for one_hot, x in molecule_list:
        atom_type = one_hot.argmax(2).squeeze(0).cpu().detach().numpy()
        x = x.squeeze(0).cpu().detach().numpy()

        validity_results = check_stability(x, atom_type)

        molecule_stable += int(validity_results[0])
        nr_stable_bonds += int(validity_results[1])
        n_atoms += int(validity_results[2])

        if validity_results[0]:
            molecule_stable_list.append((x, atom_type))

    # Validity
    fraction_mol_stable = molecule_stable / float(n_samples)
    fraction_atm_stable = nr_stable_bonds / float(n_atoms)
    validity_dict = {
        'mol_stable': fraction_mol_stable,
        'atm_stable': fraction_atm_stable,
    }

    #print('Validity:', validity_dict)

    return validity_dict, molecule_stable_list

def analyze_node_distribution(mol_list, save_path):
    hist_nodes = Histogram_discrete('Histogram # nodes (stable molecules)')
    hist_atom_type = Histogram_discrete('Histogram of atom types')

    for molecule in mol_list:
        positions, atom_type = molecule
        hist_nodes.add([positions.shape[0]])
        hist_atom_type.add(atom_type)
    print("Histogram of #nodes")
    print(hist_nodes.bins)
    print("Histogram of # atom types")
    print(hist_atom_type.bins)
    hist_nodes.normalize()


if __name__ == '__main__':
    matplotlib.use('macosx')

    main_analyze_qm9()
    # main_check_stability()

