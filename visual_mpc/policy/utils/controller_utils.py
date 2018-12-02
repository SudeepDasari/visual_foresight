import numpy as np


def truncate_movement(actions, hp):
    maxshift = hp.initial_std * 2

    if len(actions.shape) == 3:
        if hp.action_order[0] is not None:
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
        if hp.action_order[0] is not None:
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

    if hp.action_order[0] is not None:
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
