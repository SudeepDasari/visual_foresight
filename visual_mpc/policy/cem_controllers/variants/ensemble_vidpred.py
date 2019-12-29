from ..pixel_cost_controller import PixelCostController
import numpy as np
import pdb


class CEM_Controller_Ensemble_Vidpred(PixelCostController):
    def __init__(self, ag_params, policyparams, gpu_id, ngpu):
        super(CEM_Controller_Ensemble_Vidpred, self).__init__(ag_params, policyparams, gpu_id, ngpu)
        self._ens_block_len = self._net_bsize // self._hp.num_ensembles
        self._n_ensemble_blocks = self.M // self._ens_block_len

    def _default_hparams(self):
        default_params = super(CEM_Controller_Ensemble_Vidpred, self)._default_hparams()
        default_params.add_hparam('num_ensembles', 4)
        default_params.add_hparam('lambda_variance', 0.1)
        return default_params

    def evaluate_rollouts(self, actions, cem_itr, itr_times, n_samps=None):
        repeated_actions = np.zeros((self.M * self._hp.num_ensembles,
                                     actions.shape[1], actions.shape[2]), dtype=actions.dtype)

        for i in range(self._n_ensemble_blocks):
            start_ind = self._net_bsize * i
            to_repeat = actions[self._ens_block_len * i: self._ens_block_len * (i + 1)]
            for j in range(self._hp.num_ensembles):
                repeated_actions[start_ind + j * self._ens_block_len: start_ind + (j + 1) * self._ens_block_len] = to_repeat.copy()

        ensemble_scores = super(CEM_Controller_Ensemble_Vidpred, self).evaluate_rollouts(repeated_actions, cem_itr,
                                                                                         itr_times, self.M * self._hp.num_ensembles)
        return ensemble_scores

    def _expected_distance(self, icam, idesig, gen_distrib, distance_grid, normalize=True):
        """
        :param gen_distrib: shape [batch, t, r, c]
        :param distance_grid: shape [r, c]
        :return:
        """
        assert len(gen_distrib.shape) == 4
        t_mult = np.ones([self._net_seqlen - self.netconf['context_frames']])
        t_mult[-1] = self._hp.finalweight

        gen_distrib = gen_distrib.copy()
        #normalize prob distributions
        if normalize:
            gen_distrib /= np.sum(np.sum(gen_distrib, axis=2), 2)[:,:, None, None]
        gen_distrib *= distance_grid[None, None]
        scores = np.sum(np.sum(gen_distrib, axis=2),2)
        ens_scores = [[] for _ in range(self._hp.num_ensembles)]
        for i in range(self._n_ensemble_blocks):
            start_ind = self._net_bsize * i
            for j in range(self._hp.num_ensembles):
                ens_scores[j].append(scores[start_ind + j * self._ens_block_len: start_ind + (j + 1) * self._ens_block_len])
        for j in range(self._hp.num_ensembles):
            ens_scores[j] = np.concatenate(ens_scores[j], 0)

        mean_score = np.mean(ens_scores, axis=0)
        variance_score = np.var(ens_scores, axis=0)
        scores = mean_score + self._hp.lambda_variance * variance_score
        self.cost_perstep[:,icam, idesig] = scores
        scores *= t_mult[None]
        scores = np.sum(scores, axis=1)/np.sum(t_mult)
        return scores