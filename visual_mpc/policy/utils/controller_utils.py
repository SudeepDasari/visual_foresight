import numpy as np
import copy
import sys


def truncate_movement(actions, hp):
    maxshift = hp.initial_std * 2

    if len(actions.shape) == 3:
        if hp.action_order is not None:
            for i, a in enumerate(hp.action_order):
                if a == 'x' or a == 'y':
                    maxshift = hp.initial_std * 2
                elif a == 'theta':
                    maxshift = np.pi / 4
                else:
                    continue
                actions[:, :, i] = np.clip(actions[:, :, i], -maxshift, maxshift)
            return actions

        actions[:,:,:2] = np.clip(actions[:,:,:2], -maxshift, maxshift)  # clip in units of meters
        if actions.shape[-1] >= 4: # if rotation is enabled
            maxrot = np.pi / 4
            actions[:, :, 3] = np.clip(actions[:, :, 3], -maxrot, maxrot)

    elif len(actions.shape) == 2:
        if hp.action_order is not None:
            for i, a in enumerate(hp.action_order):
                if a == 'x' or a == 'y':
                    maxshift = hp.initial_std * 2
                elif a == 'theta':
                    maxshift = np.pi / 4
                else:
                    continue
                actions[:, i] = np.clip(actions[:, i], -maxshift, maxshift)
            return actions

        actions[:,:2] = np.clip(actions[:,:2], -maxshift, maxshift)  # clip in units of meters
        if actions.shape[-1] >= 4: # if rotation is enabled
            maxrot = np.pi / 4
            actions[:, 3] = np.clip(actions[:, 3], -maxrot, maxrot)
    else:
        raise NotImplementedError
    return actions


def construct_initial_sigma(hp, adim, t=None):
    xy_std = hp.initial_std
    diag = [xy_std**2, xy_std**2]

    if hp.action_order is not None:
        diag = []
        for a in hp.action_order:
            if a == 'x' or a == 'y':
                diag.append(xy_std**2)
            elif a == 'z':
                diag.append(hp.initial_std_lift ** 2)
            elif a == 'theta':
                diag.append(hp.initial_std_rot ** 2)
            elif a == 'grasp':
                diag.append(hp.initial_std_grasp ** 2)
            else:
                raise NotImplementedError
    else:
        if adim >= 3:
            diag.append(hp.initial_std_lift ** 2)
        if adim >= 4:
            diag.append(hp.initial_std_rot ** 2)
        if adim == 5:
            diag.append(hp.initial_std_grasp ** 2)

    adim = len(diag)
    diag = np.tile(diag, hp.nactions)
    diag = np.array(diag)

    if 'reduce_std_dev' in hp:
        assert 'reuse_mean' in hp
        if t >= 2:
            print('reducing std dev by factor', hp.reduce_std_dev)
            # reducing all but the last repeataction in the sequence since it can't be reused.
            diag[:(hp.nactions - 1) * adim] *= hp.reduce_std_dev

    sigma = np.diag(diag)
    return sigma


def reuse_cov(sigma, adim, hp):
    assert hp.replan_interval == 3
    print('reusing cov form last MPC step...')
    sigma_old = copy.deepcopy(sigma)
    sigma = np.zeros_like(sigma)
    #reuse covariance and add a fraction of the initial covariance to it
    sigma[0:-adim,0:-adim] = sigma_old[adim:,adim: ] + \
                             construct_initial_sigma(hp, adim)[:-adim, :-adim] * hp.reuse_cov
    sigma[-adim:, -adim:] = construct_initial_sigma(hp, adim)[:adim, :adim]
    return sigma


def make_blockdiagonal(cov, nactions, adim):
    mat = np.zeros_like(cov)
    for i in range(nactions-1):
        mat[i*adim:i*adim + adim*2, i*adim:i*adim + adim*2] = np.ones([adim*2, adim*2])
    newcov = cov*mat
    return newcov


def discretize(actions, M, naction_steps, discrete_ind):
    """
    discretize and clip between 0 and 4
    :param actions:
    :return:
    """
    for b in range(M):
        for a in range(naction_steps):
            for ind in discrete_ind:
                actions[b, a, ind] = np.clip(np.floor(actions[b, a, ind]), 0, 4)
    return actions
