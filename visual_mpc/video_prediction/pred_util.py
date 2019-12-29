import numpy as np


def get_context(n_context, t, state, images, hp=None):
    last_frames = images[t - n_context + 1:t + 1]  # same as [t - 1:t + 1] for context 2
    last_frames = last_frames.astype(np.float32, copy=False) / 255.
    last_frames = last_frames[None]
    last_states = state[t - n_context + 1:t + 1]
    last_states = last_states[None]
    if hp and hp.state_append:
        last_state_append = np.tile(np.array([[hp.state_append]]), (1, n_context, 1))
        last_states = np.concatenate((last_states, last_state_append), -1)
    return last_frames, last_states


def _check_and_slice(arr, batch_size):
    if arr is not None:
        return arr[:batch_size]


def rollout_predictions(predictor, b_size, actions, context_frames, context_states=None, input_distribs=None, logger=None):
    num_actions = actions.shape[0]
    # -(-a // b) is ceil in python
    nruns = max(1, -(-num_actions // b_size))

    gen_images, gen_distrib, gen_state = [], [], []
    for run in range(nruns):
        action_batch = actions[run * b_size:(run + 1) * b_size]
        if run == nruns - 1:
            T, adim = action_batch.shape[1:]
            padded_action_batch = np.zeros((b_size, T, adim))
            padded_action_batch[:action_batch.shape[0]] = action_batch
        else:
            padded_action_batch = action_batch

        if logger:
            logger.log("Vpred run: {} with {} actions".format(run, action_batch.shape[0]))

        _gen_images, _gen_distrib, _gen_state = predictor(input_images=context_frames,
                                                       input_state=context_states,
                                                       input_actions=padded_action_batch,
                                                       input_one_hot_images=input_distribs)

        gen_images.append(_check_and_slice(_gen_images, action_batch.shape[0]))
        gen_distrib.append(_check_and_slice(_gen_distrib, action_batch.shape[0]))
        gen_state.append(_check_and_slice(_gen_state, action_batch.shape[0]))

    return gen_images, gen_distrib, gen_state
