import argparse
import torch
import utils
import wandb
from dw4_experiment import losses
from dw4_experiment.dataset import get_data
from dw4_experiment.models import get_model
from flows.distributions import PositionPrior
from flows.utils import remove_mean


parser = argparse.ArgumentParser(description='SE3')
parser.add_argument('--model', type=str, default='simple_dynamics',
                    help='our_dynamics | schnet | simple_dynamics | kernel_dynamics | egnn_dynamics | gnn_dynamics')
parser.add_argument('--data', type=str, default='lj13',
                    help='dw4 | lj13')
parser.add_argument('--n_epochs', type=int, default=300)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--n_data', type=int, default=1000,
                    help="Number of training samples")
parser.add_argument('--sweep_n_data', type=eval, default=False,
                    help="sweep n_data instead of using the provided parameter")
parser.add_argument('--condition_time', type=eval, default=True,
                    help='True | False')
parser.add_argument('--trace', type=str, default='hutch',
                    help='hutch | exact')
parser.add_argument('--tanh', type=eval, default=True,
                    help='use tanh in the coord_mlp')
parser.add_argument('--hutch_noise', type=str, default='bernoulli',
                    help='gaussian | bernoulli')
parser.add_argument('--nf', type=int, default=32,
                    help='number of layers')
parser.add_argument('--n_layers', type=int, default=3,
                    help='number of layers')
parser.add_argument('--name', type=str, default='debug')
parser.add_argument('--wandb_usr', type=str, default='')
parser.add_argument('--n_report_steps', type=int, default=1)
parser.add_argument('--test_epochs', type=int, default=2)
parser.add_argument('--attention', type=eval, default=True,
                    help='use attention in the EGNN')
parser.add_argument('--data_augmentation', type=eval, default=False,
                    help='use data augmentation')
parser.add_argument('--weight_decay', type=float, default=1e-12,
                    help='use data augmentation')
parser.add_argument('--ode_regularization', type=float, default=0)
parser.add_argument('--x_aggregation', type=str, default='sum',
                    help='sum | mean')

args, unparsed_args = parser.parse_known_args()
if args.model == 'kernel_dynamics' and args.data == 'lj13':
    args.model = 'kernel_dynamics_lj13'
print(args)

if args.data == 'dw4':
    n_particles = 4
    n_dims = 2
    dim = n_particles * n_dims  # system dimensionality
elif args.data == 'lj13':
    n_particles = 13
    n_dims = 3
    dim = n_particles * n_dims
else:
    raise Exception('wrong data partition: %s' % args.data)


def main():
    prior = PositionPrior()  # set up prior

    flow = get_model(args, dim, n_particles)
    if torch.cuda.is_available():
        flow = flow.cuda()

    # Log all args to wandb
    wandb.init(entity=args.wandb_usr, project='se3flows', name=args.name, config=args)
    wandb.save('*.txt')
    # logging.write_info_file(model=dynamics, FLAGS=args,
    #                         UNPARSED_ARGV=unparsed_args,
    #                         wandb_log_dir=wandb.run.dir)

    data_train, batch_iter_train = get_data(args, 'train', args.batch_size)
    data_val, batch_iter_val = get_data(args, 'val', batch_size=100)
    data_test, batch_iter_test = get_data(args, 'test', batch_size=100)

    print("Max")
    print(torch.max(data_train))

    # initial training with likelihood maximization on data set
    optim = torch.optim.AdamW(flow.parameters(), lr=args.lr, amsgrad=True,
                              weight_decay=args.weight_decay)
    print(flow)

    best_val_loss = 1e8
    best_test_loss = 1e8
    for epoch in range(args.n_epochs):
        nll_epoch = []
        flow.set_trace(args.trace)
        for it, idxs in enumerate(batch_iter_train):
            batch = data_train[idxs]
            assert batch.size(0) == args.batch_size

            batch = batch.view(batch.size(0), n_particles, n_dims)
            batch = remove_mean(batch)

            if args.data_augmentation:
                batch = utils.random_rotation(batch).detach()

            if torch.cuda.is_available():
                batch = batch.cuda()

            optim.zero_grad()

            # transform batch through flow


            if 'kernel_dynamics' in args.model:
                loss, nll, reg_term, mean_abs_z = losses.compute_loss_and_nll_kerneldynamics(args, flow, prior, batch, n_particles, n_dims)
            else:
                loss, nll, reg_term, mean_abs_z = losses.compute_loss_and_nll(args, flow, prior, batch)
            # standard nll from forward KL

            loss.backward()

            optim.step()

            if it % args.n_report_steps == 0:
                print("\repoch: {0}, iter: {1}/{2}, NLL: {3:.4} Reg term: {4:.3f}".format(
                    epoch,
                    it,
                    len(batch_iter_train),
                    nll.item(),
                    reg_term.item()
                ))

            nll_epoch.append(nll.item())

            # wandb.log({"mean(abs(z))": mean_abs_z}, commit=False)
            # wandb.log({"Batch NLL": nll.item()}, commit=True)

        # wandb.log({"Train Epoch NLL": np.mean(nll_epoch)}, commit=False)

        if epoch % args.test_epochs == 0:
            val_loss = test(args, data_val, batch_iter_val, flow, prior, epoch, partition='val')
            test_loss = test(args, data_test, batch_iter_test, flow, prior, epoch, partition='test')
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_test_loss = test_loss
            print('Best val loss: %.4f \t Best test loss:  %.4f' % (best_val_loss, best_test_loss))

        print()  # Clear line

    return best_test_loss


def test(args, data_test, batch_iter_test, flow, prior, epoch, partition='test'):
    # use OTD in the evaluation process
    flow._use_checkpoints = False
    if args.data == 'dw4':
        flow.set_trace('exact')

    print('Testing %s partition ...' % partition)
    data_nll = 0.
    # batch_iter = BatchIterator(len(data_smaller), n_batch)
    with torch.no_grad():
        for it, batch_idxs in enumerate(batch_iter_test):
            batch = torch.Tensor(data_test[batch_idxs])
            if torch.cuda.is_available():
                batch = batch.cuda()
            batch = batch.view(batch.size(0), n_particles, n_dims)
            if 'kernel_dynamics' in args.model:
                loss, nll, reg_term, mean_abs_z = losses.compute_loss_and_nll_kerneldynamics(args, flow, prior, batch,
                                                                                             n_particles, n_dims)
            else:
                loss, nll, reg_term, mean_abs_z = losses.compute_loss_and_nll(args, flow, prior, batch)
            print("\r{}".format(it), nll, end="")
            data_nll += nll.item()
        data_nll = data_nll / (it + 1)

        print()
        print(f'%s nll {data_nll}' % partition)
        wandb.log({"Test NLL": data_nll}, commit=False)

        # TODO: no evaluation on hold out data yet
    flow.set_trace(args.trace)
    return data_nll

if __name__ == "__main__":
    if args.sweep_n_data:
        optimal_losses = []

        if args.data == "dw4":
            sweep_n_data = [100, 1000, 10000, 100000]
            epochs = [1000, 300, 50, 6]
            test_epochs = [20, 5, 1, 1]
            batch_sizes = [100, 100, 100, 100]
        elif args.data == "lj13":
            sweep_n_data = [10, 100, 1000, 10000]
            epochs = [500, 1000, 300, 50]
            test_epochs = [10, 25, 5, 1]
            batch_sizes = [10, 100, 100, 100]
        else:
            raise Exception("Not working")
        for n_data, n_epochs, test_epoch, batch_size in zip(sweep_n_data, epochs, test_epochs, batch_sizes):
            args.n_data = n_data
            args.n_epochs = n_epochs
            args.test_epochs = test_epoch
            args.batch_size = batch_size
            print("\n###########################" +
                  ("\nSweeping experiment, number of training samples --> %d \n" % n_data) +
                  "###########################\n")
            nll_loss = main()
            optimal_losses.append(nll_loss)
            print("Optimal losses")
            print(optimal_losses)
            print("\n###########################" +
                  ("\nOptimal test loss for %d training samples --> %.4f \n" % (n_data, nll_loss)) +
                  "###########################\n")
    else:
        main()


