from .general_agent import GeneralAgent
import numpy as np

class OfflineAgent(GeneralAgent):
    def __init__(self, hyperparams):
        super().__init__(hyperparams)

    def _post_process_obs(self, env_obs, agent_data, initial_obs=False):
        # obs = super()._post_process_obs(env_obs, agent_data, initial_obs=False)
        obs = {}
        images = self._goal_image[:, None]  # add dimensions for ncam
        obs['images'][:2,:] = images
        obs['states'] = np.zeros(images.shape[0], 1)  # dummy states
