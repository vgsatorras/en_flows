import numpy as np
import torch
from flows.utils import assert_mean_zero_with_mask, remove_mean_with_mask,\
    assert_correctly_masked
from qm9.analyze import check_stability


def rotate_chain(z):
    assert z.size(0) == 1

    z_h = z[:, :, 3:]

    n_steps = 30
    theta = 0.6 * np.pi / n_steps
    Qz = torch.tensor(
        [[np.cos(theta), -np.sin(theta), 0.],
         [np.sin(theta), np.cos(theta), 0.],
         [0., 0., 1.]]
    ).float()
    Qx = torch.tensor(
        [[1., 0., 0.],
         [0., np.cos(theta), -np.sin(theta)],
         [0., np.sin(theta), np.cos(theta)]]
    ).float()
    Qy = torch.tensor(
        [[np.cos(theta), 0., np.sin(theta)],
         [0., 1., 0.],
         [-np.sin(theta), 0., np.cos(theta)]]
    ).float()

    Q = torch.mm(torch.mm(Qz, Qx), Qy)

    Q = Q.to(z.device)

    results = []
    results.append(z)
    for i in range(n_steps):
        z_x = results[-1][:, :, :3]
        # print(z_x.size(), Q.size())
        new_x = torch.matmul(z_x.view(-1, 3), Q.T).view(1, -1, 3)
        # print(new_x.size())
        new_z = torch.cat([new_x, z_h], dim=2)
        results.append(new_z)

    results = torch.cat(results, dim=0)
    return results


def sample_chain(args, device, flow, dequantizer, prior, n_tries, n_nodes=19):
    flow.set_trace('exact')
    n_samples = 1

    # TODO FIX: This conditioning just zeros.
    if args.context_node_nf > 0:
        context = torch.zeros(n_samples, n_nodes, args.context_node_nf).to(device)
    else:
        context = None

    node_mask = torch.ones(n_samples, n_nodes, 1).to(device)

    edge_mask = (1 - torch.eye(n_nodes)).unsqueeze(0)
    edge_mask = edge_mask.repeat(n_samples, 1, 1).view(-1, 1).to(device)

    for i in range(n_tries):
        z_x, z_h = prior.sample(n_samples, n_nodes, node_mask)
        z = torch.cat([z_x, z_h], dim=2)
        tmp = flow.reverse(z, node_mask, edge_mask, context)
        x = tmp[:, :, 0:3]
        one_hot = tmp[:, :, 3:8]
        charges = tmp[:, :, 8:]
        tensor = dequantizer.reverse(
            {'categorical': one_hot, 'integer': charges})
        one_hot, charges = tensor['categorical'], tensor['integer']
        atom_type = one_hot.argmax(2).squeeze(0).cpu().detach().numpy()
        x_squeeze = x.squeeze(0).cpu().detach().numpy()
        mol_stable = check_stability(x_squeeze, atom_type)[0]

        if mol_stable:
            print('Found stable molecule to visualize :)')
            break
        elif i == n_tries - 1:
            print('Did not find stable molecule, showing last sample.')

    # DO NOT plot prior rotations, actnorm is missing.
    prior_rotations = rotate_chain(z)

    assert_mean_zero_with_mask(z_x, node_mask)

    zs = flow.reverse_chain(z, node_mask, edge_mask, context)
    zs_rotated = flow.reverse_chain(prior_rotations[-1:], node_mask, edge_mask, context)

    data_rotations = rotate_chain(zs[-1:])

    # This is the one we have to plot ;)
    prior_rotations_with_actnorm = rotate_chain(zs[0:1])

    def reverse_tensor(x):
        return x[torch.arange(x.size(0) - 1, -1, -1)]

    zs_rotated_rev = reverse_tensor(zs_rotated)
    prior_rotations_rev = reverse_tensor(prior_rotations_with_actnorm)

    z = torch.cat(
        [zs, data_rotations, zs_rotated_rev, prior_rotations_rev], dim=0
    )

    x = z[:, :, 0:3]
    one_hot = z[:, :, 3:8]
    charges = z[:, :, 8:]
    tensor = dequantizer.reverse({'categorical': one_hot, 'integer': charges})

    one_hot, charges = tensor['categorical'], tensor['integer']

    flow.set_trace(args.trace)
    return one_hot, charges, x


def sample(args, device, flow, dequantizer, prior, n_samples=5, n_nodes=10):
    flow.set_trace('exact')

    # TODO FIX: This conditioning just zeros.
    if args.context_node_nf > 0:
        context = torch.zeros(n_samples, n_nodes, args.context_node_nf).to(device)
    else:
        context = None

    node_mask = torch.ones(n_samples, n_nodes, 1).to(device)

    z_x, z_h = prior.sample(n_samples, n_nodes, node_mask)
    z = torch.cat([z_x, z_h], dim=2)

    assert_mean_zero_with_mask(z_x, node_mask)

    edge_mask = (1 - torch.eye(n_nodes)).unsqueeze(0)
    edge_mask = edge_mask.repeat(n_samples, 1, 1).view(-1, 1).to(device)
    z = flow.reverse(z, node_mask, edge_mask, context)

    if torch.any(torch.isnan(z)).item() or torch.any(torch.isinf(z)).item():
        print('NaN occured, setting z to zero.')
        z = torch.zeros_like(z)

    assert_correctly_masked(z, node_mask)

    x = z[:, :, 0:3]
    one_hot = z[:, :, 3:8]
    charges = z[:, :, 8:]

    assert_mean_zero_with_mask(x, node_mask)

    tensor = dequantizer.reverse({'categorical': one_hot, 'integer': charges})
    one_hot, charges = tensor['categorical'], tensor['integer']
    flow.set_trace(args.trace)
    return one_hot, charges, x
