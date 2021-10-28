import utils
import argparse
import wandb
from os.path import join
from qm9 import dataset
from qm9 import losses
from qm9.models import get_optim, get_model
from flows.utils import assert_mean_zero_with_mask, remove_mean_with_mask,\
    assert_correctly_masked
import torch
import time
import pickle
import numpy as np
import qm9.visualizer as vis
from qm9.analyze import analyze_stability_for_molecules
from qm9.utils import prepare_context
from qm9.sampling import sample_chain, sample


parser = argparse.ArgumentParser(description='SE3')
parser.add_argument('--exp_name', type=str, default='debug_10')
parser.add_argument('--model', type=str, default='egnn_dynamics',
                    help='our_dynamics | schnet | simple_dynamics | '
                         'kernel_dynamics | egnn_dynamics |gnn_dynamics')
parser.add_argument('--n_epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--brute_force', type=eval, default=False,
                    help='True | False')
parser.add_argument('--actnorm', type=eval, default=True,
                    help='True | False')
parser.add_argument('--break_train_epoch', type=eval, default=False,
                    help='True | False')
parser.add_argument('--dp', type=eval, default=True,
                    help='True | False')
parser.add_argument('--condition_time', type=eval, default=True,
                    help='True | False')
parser.add_argument('--clip_grad', type=eval, default=True,
                    help='True | False')
parser.add_argument('--trace', type=str, default='hutch',
                    help='hutch | exact')
parser.add_argument('--n_layers', type=int, default=6,
                    help='number of layers')
parser.add_argument('--nf', type=int, default=64,
                    help='number of layers')
parser.add_argument('--ode_regularization', type=float, default=1e-3)
parser.add_argument('--dataset', type=str, default='qm9',
                    help='qm9 | qm9_positional')
parser.add_argument('--dequantization', type=str, default='argmax_variational',
                    help='uniform | variational | argmax_variational')
parser.add_argument('--tanh', type=eval, default=True,
                    help='use tanh in the coord_mlp')
parser.add_argument('--attention', type=eval, default=True,
                    help='use attention in the EGNN')
parser.add_argument('--n_report_steps', type=int, default=1)
parser.add_argument('--wandb_usr', type=str, default='')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--save_model', type=eval, default=True,
                    help='save model')
parser.add_argument('--generate_epochs', type=int, default=1,
                    help='save model')
parser.add_argument('--num_workers', type=int, default=0, help='Number of worker for the dataloader')
parser.add_argument('--test_epochs', type=int, default=1)
parser.add_argument('--data_augmentation', type=eval, default=False,
                    help='use attention in the EGNN')
parser.add_argument('--x_aggregation', type=str, default='sum',
                    help='sum | mean')
parser.add_argument("--conditioning", nargs='+', default=[],
                    help='multiple arguments can be passed, '
                         'including: homo | onehot | lumo | num_atoms | etc. '
                         'usage: "--conditioning H_thermo homo onehot H_thermo"')

parser.add_argument('--resume', type=str, default=None,
                    help='')
parser.add_argument('--start_epoch', type=int, default=0,
                    help='')

args, unparsed_args = parser.parse_known_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
dtype = torch.float32

if args.resume is not None:
    exp_name = args.exp_name + '_resume'
    start_epoch = args.start_epoch
    resume = args.resume
    wandb_usr = args.wandb_usr

    with open(join(args.resume, 'args.pickle'), 'rb') as f:
        args = pickle.load(f)

    args.resume = resume
    args.break_train_epoch = False

    args.exp_name = exp_name
    args.start_epoch = start_epoch
    args.wandb_usr = wandb_usr

    print(args)

utils.create_folders(args)
print(args)


# Log all args to wandb
wandb.init(entity=args.wandb_usr, project='se3flows_qm9', name=args.exp_name, config=args)
wandb.save('*.txt')

# Retrieve QM9 dataloaders
dataloaders, charge_scale = dataset.retrieve_dataloaders(args.batch_size, args.num_workers)

data_dummy = next(iter(dataloaders['train']))


if len(args.conditioning) > 0:
    print(f'Conditioning on {args.conditioning}')
    context_dummy = prepare_context(args.conditioning, data_dummy)
    context_node_nf = context_dummy.size(2)
else:
    context_node_nf = 0

args.context_node_nf = context_node_nf


# Create EGNN flow
prior, flow, dequantizer, nodes_dist = get_model(args, device)
flow = flow.to(device)
dequantizer = dequantizer.to(device)
optim = get_optim(args, flow, dequantizer)
print(flow)

gradnorm_queue = utils.Queue()
gradnorm_queue.add(3000)  # Add large value that will be flushed.


def check_mask_correct(variables, node_mask):
    for variable in variables:
        assert_correctly_masked(variable, node_mask)


def train_epoch(loader, epoch, flow, flow_dp):
    nll_epoch = []
    for i, data in enumerate(loader):
        # Get data
        x = data['positions'].to(device, dtype)
        node_mask = data['atom_mask'].to(device, dtype).unsqueeze(2)
        edge_mask = data['edge_mask'].to(device, dtype)
        one_hot = data['one_hot'].to(device, dtype)
        charges = data['charges'].to(device, dtype).unsqueeze(2)

        x = remove_mean_with_mask(x, node_mask)
        if args.data_augmentation:
            x = utils.random_rotation(x).detach()

        check_mask_correct([x, one_hot, charges], node_mask)
        assert_mean_zero_with_mask(x, node_mask)

        h = {'categorical': one_hot, 'integer': charges}

        if len(args.conditioning) > 0:
            context = prepare_context(args.conditioning, data).to(device, dtype)
            assert_correctly_masked(context, node_mask)
        else:
            context = None

        optim.zero_grad()

        # transform batch through flow
        nll, reg_term, mean_abs_z = losses.compute_loss_and_nll(args, dequantizer, flow_dp, prior, nodes_dist, x, h,
                                                                node_mask, edge_mask, context)
        # standard nll from forward KL

        loss = nll + args.ode_regularization * reg_term

        loss.backward()

        if args.clip_grad:
            grad_norm = utils.gradient_clipping(flow, gradnorm_queue)
        else:
            grad_norm = 0.

        optim.step()

        if i % args.n_report_steps == 0:
            print(f"\repoch: {epoch}, iter: {i}/{len(loader)}, "
                  f"Loss {loss.item():.2f}, NLL: {nll.item():.2f}, "
                  f"RegTerm: {reg_term.item():.1f}, "
                  f"GradNorm: {grad_norm:.1f}")

        nll_epoch.append(nll.item())

        if i % 100 == 0:
            save_and_sample_chain(epoch=epoch)
            sample_different_sizes_and_save(epoch=epoch)
            vis.visualize("outputs/%s/epoch_%d" % (args.exp_name, epoch), wandb=wandb)
            vis.visualize_chain(
                "outputs/%s/epoch_%d/chain/" % (args.exp_name, epoch),
                wandb=wandb)

        wandb.log({"mean(abs(z))": mean_abs_z}, commit=False)
        wandb.log({"Batch NLL": nll.item()}, commit=True)

        if args.break_train_epoch:
            break

    wandb.log({"Train Epoch NLL": np.mean(nll_epoch)}, commit=False)


def test(loader, epoch, flow_dp, partition='Test'):
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
            nll, _, _ = losses.compute_loss_and_nll(args, dequantizer, flow_dp, prior, nodes_dist, x, h, node_mask,
                                                    edge_mask, context)
            # standard nll from forward KL

            nll_epoch += nll.item() * batch_size
            n_samples += batch_size
            if i % args.n_report_steps == 0:
                print(f"\r {partition} NLL \t epoch: {epoch}, iter: {i}/{len(loader)}, "
                      f"NLL: {nll_epoch/n_samples:.2f}")

            if args.break_train_epoch:
                break

    return nll_epoch/n_samples


def save_and_sample_chain(epoch=0, id_from=0):
    one_hot, charges, x = sample_chain(
        args, device, flow, dequantizer, prior, n_tries=1)

    vis.save_xyz_file(
        'outputs/%s/epoch_%d/chain/' % (args.exp_name, epoch), one_hot, charges, x,
        id_from, name='chain')

    return one_hot, charges, x


def sample_different_sizes_and_save(n_samples=10, epoch=0):
    for counter in range(n_samples):
        n_nodes = nodes_dist.sample()
        one_hot, charges, x = sample(args, device, flow, dequantizer, prior, n_samples=1, n_nodes=n_nodes)

        vis.save_xyz_file(
            'outputs/%s/epoch_%d/' % (args.exp_name, epoch), one_hot,
            charges, x,
            1*counter, name='molecule')


def analyze_and_save(epoch, n_samples=1000):
    print('Analyzing molecule validity...')
    molecule_list = []
    for i in range(n_samples):
        n_nodes = nodes_dist.sample()
        one_hot, charges, x = sample(
            args, device, flow, dequantizer, prior, n_samples=1, n_nodes=n_nodes)

        molecule_list.append((one_hot.detach(), x.detach()))

    validity_dict, _ = analyze_stability_for_molecules(molecule_list)
    wandb.log(validity_dict)
    return validity_dict


def sample_batch(prior, flow):
    print('Creating...')
    n_nodes = nodes_dist.sample()
    _, _, x = sample(args, device, flow, dequantizer, prior, n_samples=1, n_nodes=n_nodes)
    return x


def main():
    if args.resume is not None:
        flow_state_dict = torch.load(join(args.resume, 'flow.npy'))
        dequantizer_state_dict = torch.load(join(args.resume, 'dequantizer.npy'))
        optim_state_dict = torch.load(join(args.resume, 'optim.npy'))
        flow.load_state_dict(flow_state_dict)
        dequantizer.load_state_dict(dequantizer_state_dict)
        optim.load_state_dict(optim_state_dict)

    flow_dp = flow

    if args.dp and torch.cuda.device_count() > 1:
        print(f'Training using {torch.cuda.device_count()} GPUs')
        flow_dp = torch.nn.DataParallel(flow_dp.cpu())
        flow_dp = flow_dp.cuda()

    best_nll_val = 1e8
    best_nll_test = 1e8
    for epoch in range(args.start_epoch, args.n_epochs):
        start_epoch = time.time()
        train_epoch(dataloaders['train'], epoch, flow, flow_dp)
        print(f"Epoch took {time.time() - start_epoch:.1f} seconds.")

        if epoch % args.test_epochs == 0:
            analyze_and_save(epoch)
            nll_val = test(dataloaders['valid'], epoch, flow_dp, partition='Val')
            nll_test = test(dataloaders['test'], epoch, flow_dp, partition='Test')

            if nll_val < best_nll_val:
                best_nll_val = nll_val
                best_nll_test = nll_test
                if args.save_model:
                    args.current_epoch = epoch + 1
                    utils.save_model(optim, 'outputs/%s/optim.npy' % args.exp_name)
                    utils.save_model(flow, 'outputs/%s/flow.npy' % args.exp_name)
                    utils.save_model(dequantizer, 'outputs/%s/dequantizer.npy' % args.exp_name)
                    with open('outputs/%s/args.pickle' % args.exp_name, 'wb') as f:
                        pickle.dump(args, f)
            if args.save_model and epoch > 28:
                utils.save_model(optim, 'outputs/%s/optim_%d.npy' % (args.exp_name, epoch))
                utils.save_model(flow, 'outputs/%s/flow_%d.npy' % (args.exp_name, epoch))
                utils.save_model(dequantizer, 'outputs/%s/dequantizer_%d.npy' % (args.exp_name, epoch))
                with open('outputs/%s/args_%d.pickle' % (args.exp_name, epoch), 'wb') as f:
                    pickle.dump(args, f)
            print('Val loss: %.4f \t Test loss:  %.4f' % (nll_val, nll_test))
            print('Best val loss: %.4f \t Best test loss:  %.4f' % (best_nll_val, best_nll_test))
            wandb.log({"Val loss ": nll_val}, commit=True)
            wandb.log({"Test loss ": nll_test}, commit=True)
            wandb.log({"Best cross-validated test loss ": best_nll_test}, commit=True)


if __name__ == "__main__":
    main()
