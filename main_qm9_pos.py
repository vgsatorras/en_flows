import argparse
import numpy as np
import torch
import utils
import wandb
from dw4_experiment import losses
from dw4_experiment.models import get_model
from flows.distributions import PositionPrior
from flows.utils import remove_mean
from qm9 import dataset
from qm9 import analyze


parser = argparse.ArgumentParser(description='SE3')
parser.add_argument('--model', type=str, default='egnn_dynamics',
                    help='our_dynamics | schnet | simple_dynamics | kernel_dynamics | egnn_dynamics | gnn_dynamics')
parser.add_argument('--data', type=str, default='qm9_only19',
                    help='dw4 | qm9_only19')
parser.add_argument('--n_epochs', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--num_workers', type=int, default=0, help='Number of worker for the dataloader')
parser.add_argument('--condition_time', type=eval, default=True,
                    help='True | False')
parser.add_argument('--clip_grad', type=eval, default=True,
                    help='True | False')
parser.add_argument('--trace', type=str, default='hutch',
                    help='hutch | exact')
parser.add_argument('--exp_name', type=str, default='qm9pos_debug')
parser.add_argument('--wandb_usr', type=str, default='')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--n_report_steps', type=int, default=1)
parser.add_argument('--test_epochs', type=int, default=1)
parser.add_argument('--tanh', type=eval, default=True,
                    help='use tanh in the coord_mlp')
parser.add_argument('--hutch_noise', type=str, default='gaussian',
                    help='gaussian | bernoulli')
parser.add_argument('--nf', type=int, default=64,
                    help='number of layers')
parser.add_argument('--ode_regularization', type=float, default=1e-3)
parser.add_argument('--n_layers', type=int, default=6,
                    help='number of layers')
parser.add_argument('--attention', type=eval, default=True,
                    help='use attention in the EGNN')
parser.add_argument('--save_model', type=eval, default=False,
                    help='save model')
parser.add_argument('--data_augmentation', type=eval, default=False,
                    help='use attention in the EGNN')
parser.add_argument('--break_train_epoch', type=eval, default=False,
                    help='True | False')
parser.add_argument('--x_aggregation', type=str, default='sum',
                    help='sum | mean')

args, unparsed_args = parser.parse_known_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
args.device = device
dtype = torch.float32
utils.create_folders(args)
print(args)

n_particles = 19  # 19 nodes is the most common type of molecule in QM9
n_dims = 3
dim = n_dims * n_particles  # system dimensionality


def main():
    gradnorm_queue = utils.Queue()
    gradnorm_queue.add(1e30)  # Add large value that will be flushed.

    prior = PositionPrior()  # set up prior

    flow = get_model(args, dim, n_particles)
    if torch.cuda.is_available():
        flow = flow.cuda()

    # Log all args to wandb
    wandb.init(entity=args.wandb_usr, project='se3flows_qm9pos', name=args.exp_name, config=args)
    wandb.save('*.txt')

    # Retrieve QM9 dataloaders
    dataloaders, charge_scale = dataset.retrieve_dataloaders(args.batch_size, args.num_workers,
                                                             filter_n_atoms=n_particles)

    # initial training with likelihood maximization on data set
    optim = torch.optim.AdamW(flow.parameters(), lr=args.lr, amsgrad=True,
                              weight_decay=1e-12)
    print(flow)

    best_val_loss = 1e8
    best_test_loss = 1e8
    best_kl_div, best_js_div = 1e10, 1e10
    visualize_histograms(prior, flow, wandb=wandb)
    for epoch in range(args.n_epochs):
        nll_epoch = []
        flow.set_trace(args.trace)

        for it, data in enumerate(dataloaders['train']):
            batch = data['positions'].to(device, dtype)
            batch = remove_mean(batch)

            if args.data_augmentation:
                batch = utils.random_rotation(batch).detach()

            optim.zero_grad()

            # transform batch through flow
            if args.model == 'kernel_dynamics':
                loss, nll, reg_term, mean_abs_z = losses.compute_loss_and_nll_kerneldynamics(args, flow, prior, batch,
                                                                                             n_particles, n_dims)
            else:
                loss, nll, reg_term, mean_abs_z = losses.compute_loss_and_nll(args, flow, prior, batch)
            # standard nll from forward KL

            loss = nll + args.ode_regularization * reg_term

            loss.backward()

            if args.clip_grad:
                grad_norm = utils.gradient_clipping(flow, gradnorm_queue)
            else:
                grad_norm = 0.

            optim.step()

            if it % args.n_report_steps == 0:
                print("\repoch: {0}, iter: {1}/{2}, NLL: {3:.4} Reg term: {4:.3f} GradNorm: {5:.3f}".format(
                    epoch,
                    it,
                    len(dataloaders['train']),
                    nll.item(),
                    reg_term.item(),
                    grad_norm
                ))

            nll_epoch.append(nll.item())

            wandb.log({"mean(abs(z))": mean_abs_z}, commit=False)
            wandb.log({"Batch NLL": nll.item()}, commit=True)

            if args.break_train_epoch:
                break

        wandb.log({"Train Epoch NLL": np.mean(nll_epoch)}, commit=False)

        if epoch % args.test_epochs == 0:
            val_loss = test(args, dataloaders['valid'], flow, prior, epoch, partition='val')
            test_loss = test(args, dataloaders['test'], flow, prior, epoch, partition='test')
            kl_div, js_div = visualize_histograms(prior, flow, epoch=epoch, wandb=wandb)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_test_loss = test_loss
                best_kl_div = kl_div
                best_js_div = js_div
                if args.save_model:
                    utils.save_model(flow, 'outputs/%s/flow.npy' % args.exp_name)
            print('Best val loss: %.4f \t Best test loss:  %.4f' % (best_val_loss, best_test_loss))
            wandb.log({"Test loss ": test_loss}, commit=True)
            wandb.log({"Best cross-validated test loss ": best_test_loss}, commit=True)
            wandb.log({"KL div hist": kl_div}, commit=True)
            wandb.log({"Best cross-validated KL div hist": best_kl_div}, commit=True)
            wandb.log({"JS div hist": js_div}, commit=True)
            wandb.log({"Best cross-validated JS div hist": best_js_div}, commit=True)

        print()  # Clear line
    return best_test_loss


def test(args, dataloader, flow, prior, epoch, partition='test'):
    # use OTD in the evaluation process
    # flow._use_checkpoints = False
    # flow.set_trace('exact')

    print('Testing %s partition ...' % partition)
    data_nll = 0.
    # batch_iter = BatchIterator(len(data_smaller), n_batch)
    with torch.no_grad():
        for it, data in enumerate(dataloader):
            x = data['positions'].to(device, dtype)
            x = remove_mean(x)
            if torch.cuda.is_available():
                x = x.cuda()
            x = x.view(x.size(0), n_particles, 3)
            if args.model == 'kernel_dynamics':
                _, nll, _, _ = losses.compute_loss_and_nll_kerneldynamics(args, flow, prior, x, n_particles, n_dims)
            else:
                _, nll, _, _ = losses.compute_loss_and_nll(args, flow, prior, x)

            nll = nll.detach().item()
            print("\r{}".format(it), nll, end="")
            data_nll += nll
    data_nll = data_nll / (it + 1)

    print()
    print(f'Test (%s) nll {data_nll}' % partition)
    # flow.set_trace(args.trace)
    # wandb.log({"Test NLL": data_nll}, commit=False)

    # TODO: no evaluation on hold out data yet
    return data_nll


def sample_batch(prior, flow, batch_size=100):
    with torch.no_grad():
        z = prior.sample(size=(batch_size, n_particles, 3), device=device)
    return flow.reverse(z)

def visualize_histograms(prior, flow, n_samples=1000, batch_size=100, epoch=-1, wandb=None):
    flow.set_trace('exact')
    hist_dist = analyze.Histogram_cont(name="Relative distances distribution", ignore_zeros=True)
    for i in range(max(int(n_samples / batch_size), 1)):
        x_gen = sample_batch(prior, flow, batch_size)
        dist = analyze.coord2distances(x_gen).cpu().detach()
        hist_dist.add(list(dist.numpy()))

    save_path = "outputs/%s/hist_epoch_%d.png" % (args.exp_name, epoch)
    hist_dist.plot_both(analyze.analyzed_19['distances'], save_path=save_path, wandb=wandb)
    print("Histogram of bins")
    print(hist_dist.bins)
    kl_div = analyze.kl_divergence_sym(hist_dist.bins, analyze.analyzed_19['distances'])
    js_div = analyze.js_divergence(hist_dist.bins, analyze.analyzed_19['distances'])
    flow.set_trace(args.trace)
    return kl_div, js_div

if __name__ == "__main__":
    main()


