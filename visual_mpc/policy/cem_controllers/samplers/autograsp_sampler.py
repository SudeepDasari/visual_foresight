from .gaussian_sampler import GaussianCEMSampler
import numpy as np


class AutograspSampler(GaussianCEMSampler):
    def __init__(self, hp, adim, sdim, **kwargs):
        super(AutograspSampler, self).__init__(hp, adim - 1, sdim, **kwargs)
    
    @staticmethod
    def get_default_hparams():
        parent_params = GaussianCEMSampler.get_default_hparams()
        parent_params['deviation_prob'] = 0
        parent_params['reopen'] = False
        parent_params['action_norm_factor'] = 1.0                   # 100 / (high_z - low_z)
        parent_params['z_thresh'] = 0.15
        parent_params['gripper_close_cmd'] = 1
        parent_params['gripper_open_cmd'] = -1
        parent_params['no_refit'] = True
        return parent_params
    
    def sample_initial_actions(self, t, nsamples, current_state):
        self._current_state = current_state
        return self._sample_gripper(super(AutograspSampler, self).sample_initial_actions(t, nsamples, current_state), nsamples)
    
    def sample_next_actions(self, n_samples, best_actions, scores):
        default_actions = super(AutograspSampler, self).sample_next_actions(n_samples, best_actions[:, :, :-1])
        if self._hp.no_refit:
            return self._sample_gripper(default_actions, n_samples)
        
        grip_act = np.zeros((n_samples, default_actions.shape[1], 1), dtype=np.float32)
        close_prob = np.mean((best_actions[:, :, -1] == self._hp.gripper_close_cmd).astype(np.float32), axis=0)
        for t in range(default_actions.shape[1]):
            cmd_t = np.random.uniform(size=n_samples) < close_prob[t]
            grip_act[:, t, 0] = cmd_t * self._hp.gripper_close_cmd + np.logical_not(cmd_t) * self._hp.gripper_open_cmd

        return np.concatenate((default_actions, grip_act), axis=-1) 
    
    def _sample_gripper(self, default_samples, nsamples):
        grip_actions = np.zeros((nsamples, default_samples.shape[1], 1))

        for b in range(nsamples):
            close_mask = np.cumsum(default_samples[b, :, 2] * self._hp.action_norm_factor) + self._current_state[2] < self._hp.z_thresh

            if not self._hp.reopen:
                non_zeros = close_mask.nonzero()[0]
                if len(non_zeros):
                    close_mask[non_zeros[0]:] = True
            
            if self._hp.deviation_prob:
                noise_mask = np.random.uniform(size=close_mask.shape[0]) < self._hp.deviation_prob
                for t in range(close_mask.shape[0]):
                    if noise_mask[t]:
                        close_mask[t] = not close_mask[t]
            
            open_mask = np.logical_not(close_mask)
            grip_actions[b, :, 0] = open_mask * self._hp.gripper_open_cmd + close_mask * self._hp.gripper_close_cmd
        
        return np.concatenate((default_samples, grip_actions), axis=-1)

