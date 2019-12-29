""" Wraps general agent class to allow sim to easily create new task definitions """
from visual_mpc.agent.general_agent import GeneralAgent


class CreateConfigAgent(GeneralAgent):
    def __init__(self, hyperparams):
        super().__init__(hyperparams)

    def rollout(self, policy, i_trial, i_traj):

        # Take the sample.
        self._init()

        agent_data, policy_outputs = {}, []
        agent_data['traj_ok'] = True
        initial_env_obs, reset_state = self.env.reset()
        agent_data['reset_state'] = reset_state

        obs = self._post_process_obs(initial_env_obs, agent_data, initial_obs=True)

        for t in range(self._hyperparams['T']):
            self.env.generate_task()
            try:
                obs = self._post_process_obs(self.env.current_obs(), agent_data)
            except ValueError:
                return {'traj_ok': False}, None, None

        return agent_data, obs, policy_outputs