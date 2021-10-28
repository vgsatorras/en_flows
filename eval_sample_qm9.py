

import utils
import argparse
# import wandb

from qm9 import dataset
from qm9.models import get_optim, get_model

from flows.utils import assert_mean_zero_with_mask, remove_mean_with_mask,\
    assert_correctly_masked
import torch
import time
import numpy as np
import pickle
import qm9.visualizer as vis
from qm9.analyze import check_stability
from os.path import join
from qm9.sampling import sample_chain, sample


def check_mask_correct(variables, node_mask):
    for variable in variables:
        assert_correctly_masked(variable, node_mask)


def save_and_sample_chain(args, eval_args, device, flow, dequantizer, prior, n_tries, n_nodes, id_from=0):
    one_hot, charges, x = sample_chain(args, device, flow, dequantizer, prior, n_tries, n_nodes)

    vis.save_xyz_file(
        join(eval_args.model_path, 'eval/chain/'), one_hot, charges, x,
        id_from, name='chain')

    return one_hot, charges, x


def sample_different_sizes_and_save(args, eval_args, device, flow, dequantizer, prior, nodes_dist, n_samples=10):
    for counter in range(n_samples):
        n_nodes = nodes_dist.sample()
        one_hot, charges, x = sample(
            args, device, flow, dequantizer, prior, n_samples=1,
            n_nodes=n_nodes)

        vis.save_xyz_file(
            join(eval_args.model_path, 'eval/molecules/'), one_hot, charges, x,
            id_from=counter, name='molecule')


def sample_only_stable_different_sizes_and_save(args, eval_args, device, flow, dequantizer, prior, nodes_dist, n_samples=10, n_tries=1000):
    counter = 0
    for i in range(n_tries):
        if counter == n_samples:
            break

        n_nodes = nodes_dist.sample()
        one_hot, charges, x = sample(
            args, device, flow, dequantizer, prior, n_samples=1,
            n_nodes=n_nodes)

        atom_type = one_hot.argmax(2).squeeze(0).cpu().detach().numpy()
        x_squeeze = x.squeeze(0).cpu().detach().numpy()
        mol_stable = check_stability(x_squeeze, atom_type)[0]

        if mol_stable:
            print('Found stable mol.')
            vis.save_xyz_file(
                join(eval_args.model_path, 'eval/molecules/'), one_hot, charges, x,
                id_from=counter, name='molecule_stable')
            counter += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="outputs/en_flows_pretrained",
                        help='Specify model path')
    parser.add_argument('--n_tries', type=int, default=10,
                        help='N tries to find stable molecule for gif animation')
    parser.add_argument('--n_nodes', type=int, default=19,
                        help='number of atoms in molecule for gif animation')

    eval_args, unparsed_args = parser.parse_known_args()

    assert eval_args.model_path is not None

    with open(join(eval_args.model_path, 'args.pickle'), 'rb') as f:
        args = pickle.load(f)

    # CAREFUL!!!! Use to write new args to the args file.
    # with open(join(eval_args.model_path, 'args.pickle'), 'wb') as f:
    #     print('saving args.')
    #     args.nf = 64
    #     pickle.dump(args, f)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    dtype = torch.float32
    utils.create_folders(args)
    print(args)

    # Retrieve QM9 dataloaders
    dataloaders, charge_scale = dataset.retrieve_dataloaders(args.batch_size,
                                                             args.num_workers)

    data_dummy = next(iter(dataloaders['train']))

    prior, flow, dequantizer, nodes_dist = get_model(args, device)
    flow.to(device)
    dequantizer.to(device)

    flow_state_dict = torch.load(join(eval_args.model_path, 'flow.npy'), map_location=device)
    dequantizer_state_dict = torch.load(
        join(eval_args.model_path, 'dequantizer.npy'), map_location=device)

    flow.load_state_dict(flow_state_dict)
    dequantizer.load_state_dict(dequantizer_state_dict)

    print('Sampling handful of molecules and visualizing.')
    sample_different_sizes_and_save(
        args, eval_args, device, flow, dequantizer, prior, nodes_dist)
    sample_only_stable_different_sizes_and_save(
        args, eval_args, device, flow, dequantizer, prior, nodes_dist, n_samples=200)
    vis.visualize(
        join(eval_args.model_path, 'eval/molecules/'), spheres_3d=True)

    print('Sampling visualization chain.')
    save_and_sample_chain(
        args, eval_args, device, flow, dequantizer, prior,
        n_tries=eval_args.n_tries, n_nodes=eval_args.n_nodes)
    vis.visualize_chain(
        join(eval_args.model_path, 'eval/chain/'), spheres_3d=True)



if __name__ == "__main__":
    main()
