from torch.utils.data import DataLoader
from qm9.data.args import init_argparse
from qm9.data.collate import collate_fn
from qm9.data.utils import initialize_datasets

def retrieve_dataloaders(batch_size, num_workers=0, filter_n_atoms=None):
    # Initialize dataloader
    args = init_argparse('qm9')
    args, datasets, num_species, charge_scale = initialize_datasets(args, args.datadir, 'qm9',
                                                                    subtract_thermo=args.subtract_thermo,
                                                                    force_download=args.force_download
                                                                    )
    qm9_to_eV = {'U0': 27.2114, 'U': 27.2114, 'G': 27.2114, 'H': 27.2114, 'zpve': 27211.4, 'gap': 27.2114, 'homo': 27.2114,
                 'lumo': 27.2114}

    for dataset in datasets.values():
        dataset.convert_units(qm9_to_eV)

    if filter_n_atoms is not None:
        datasets = filter_atoms(datasets, filter_n_atoms)

    # Construct PyTorch dataloaders from datasets
    dataloaders = {split: DataLoader(dataset,
                                     batch_size=batch_size,
                                     shuffle=args.shuffle if (split == 'train') else False,
                                     num_workers=num_workers,
                                     collate_fn=collate_fn)
                         for split, dataset in datasets.items()}

    return dataloaders, charge_scale


def filter_atoms(datasets, n_nodes):
    for key in datasets:
        dataset = datasets[key]
        idxs = dataset.data['num_atoms'] == n_nodes
        for key2 in dataset.data:
            dataset.data[key2] = dataset.data[key2][idxs]

        datasets[key].num_pts = dataset.data['one_hot'].size(0)
        datasets[key].perm = None
    return datasets