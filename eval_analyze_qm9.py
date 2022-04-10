import utils
import argparse
from qm9 import dataset
from qm9.models import get_model
import os
from flows.utils import assert_mean_zero_with_mask, remove_mean_with_mask,\
    assert_correctly_masked
import torch
import pickle
from os.path import join
from qm9.sampling import sample
from qm9.analyze import analyze_stability_for_molecules
from qm9.utils import prepare_context
import qm9.losses as losses
from qm9 import rdkit_functions, visualizer


def check_mask_correct(variables, node_mask):
    for variable in variables:
        assert_correctly_masked(variable, node_mask)


def analyze_and_save(args, eval_args, device, flow, dequantizer, prior, nodes_dist, rdkit_bool=False, n_samples=1000):
    print('Analyzing molecule stability...')
    molecule_list = []
    for i in range(n_samples):
        n_nodes = nodes_dist.sample()
        one_hot, charges, x = sample(
            args, device, flow, dequantizer, prior, n_samples=1, n_nodes=n_nodes)

        if eval_args.save_to_xyz:
            visualizer.save_xyz_file(
                join(eval_args.model_path, 'eval/analyzed_molecules/'),
                one_hot, charges, x, i, name='molecule')

        molecule_list.append((one_hot.detach(), x.detach()))
        if i % 10 == 0:
            print('\t %d/%d Molecules generated' % (i, n_samples))

    stability_dict, molecule_stable_list = analyze_stability_for_molecules(molecule_list)

    if rdkit_bool:
        rdkit_metrics = rdkit_functions.validity_uniqueness_novelty(molecule_stable_list)
    else:
        rdkit_metrics = None

    # Histograms
    # path = join(eval_args.model_path, 'eval')
    # os.makedirs(path, exist_ok=True)
    # analyze_node_distribution(molecule_stable_list, path + "/node_dist.png")

    return stability_dict, rdkit_metrics


def test(args, dequantizer, flow_dp, prior, nodes_dist, device, dtype, loader, partition='Test'):
    with torch.no_grad():
        nll_epoch = 0
        n_samples = 0
        for i, data in enumerate(loader):
            # Get data
            x = data['positions'].to(device, dtype)
            batch_size = x.size(0)
            node_mask = data['atom_mask'].to(device, dtype).unsqueeze(2)
            edge_mask = data['edge_mask'].to(device, dtype)
            one_hot = data['one_hot'].to(device, dtype)
            charges = data['charges'].to(device, dtype).unsqueeze(2)

            x = remove_mean_with_mask(x, node_mask)
            check_mask_correct([x, one_hot, charges], node_mask)
            assert_mean_zero_with_mask(x, node_mask)

            h = {'categorical': one_hot, 'integer': charges}

            if len(args.conditioning) > 0:
                context = prepare_context(args.conditioning, data).to(device, dtype)
                assert_correctly_masked(context, node_mask)
            else:
                context = None

            # transform batch through flow
            nll, _, _ = losses.compute_loss_and_nll(args, dequantizer, flow_dp, prior, nodes_dist, x, h, node_mask, edge_mask, context)
            # standard nll from forward KL

            nll_epoch += nll.item() * batch_size
            n_samples += batch_size
            if i % args.n_report_steps == 0:
                print(f"\r {partition} NLL \t, iter: {i}/{len(loader)}, "
                      f"NLL: {nll_epoch/n_samples:.2f}")

    return nll_epoch/n_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="outputs/en_flows_pretrained",
                        help='Specify model path')
    parser.add_argument('--n_samples', type=int, default=10000,
                        help='Specify model path')
    parser.add_argument('--val_novel_unique', type=eval, default=True,
                        help='Whether to analyze Validity, Novelty and Uniqueness')
    parser.add_argument('--save_to_xyz', type=eval, default=False,
                        help='save xyz files')


    eval_args, unparsed_args = parser.parse_known_args()

    assert eval_args.model_path is not None

    with open(join(eval_args.model_path, 'args.pickle'), 'rb') as f:
        args = pickle.load(f)

    # CAREFUL!!!! Use to write new args to the args file.
    # with open(join(eval_args.model_path, 'args.pickle'), 'wb') as f:
    #     print('saving args.')
    #     args.x_aggregation = 'sum'
    #     pickle.dump(args, f)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    dtype = torch.float32
    utils.create_folders(args)
    print(args)

    # Retrieve QM9 dataloaders
    dataloaders, charge_scale = dataset.retrieve_dataloaders(args.batch_size,
                                                             args.num_workers)

    # Load model
    prior, flow, dequantizer, nodes_dist = get_model(args, device)
    flow.to(device)
    dequantizer.to(device)
    flow_state_dict = torch.load(join(eval_args.model_path, 'flow.npy'), map_location=device)
    dequantizer_state_dict = torch.load(
        join(eval_args.model_path, 'dequantizer.npy'), map_location=device)
    flow.load_state_dict(flow_state_dict)
    dequantizer.load_state_dict(dequantizer_state_dict)

    # Analyze stability, validity, uniqueness and novelty
    stability_dict, rdkit_metrics = analyze_and_save(
        args, eval_args, device, flow, dequantizer, prior, nodes_dist, eval_args.val_novel_unique,
        n_samples=eval_args.n_samples)
    print(stability_dict)
    print(rdkit_metrics)

    # Evaluate negative log-likelihood for the validation and test partitions
    val_nll = test(args, dequantizer, flow, prior, nodes_dist, device, dtype,
                   dataloaders['valid'],
                   partition='Val')
    print(f'Final val nll {val_nll}')
    test_nll = test(args, dequantizer, flow, prior, nodes_dist, device, dtype,
                    dataloaders['test'],
                    partition='Test')
    print(f'Final test nll {test_nll}')
    print(f'Overview: val nll {val_nll} test nll {test_nll}', stability_dict)
    with open(join(eval_args.model_path, 'eval/log.txt'), 'w') as f:
        print(f'Overview: val nll {val_nll} test nll {test_nll}', stability_dict,
              file=f)


if __name__ == "__main__":
    main()
