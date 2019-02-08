#!/usr/bin/env python
import os
import argparse
import imp
import cPickle as pkl
import numpy as np
import shutil
import cv2
from visual_mpc.envs.base_env import supported_robots


class RobotEnvironment:
    def __init__(self, robot_name, conf, resume=False, ngpu=1, gpu_id=0, is_bench=False):
        if robot_name not in supported_robots:
            msg = "ROBOT {} IS NOT A SUPPORTED ROBOT (".format(robot_name)
            for k in supported_robots:
                msg = msg + "{} ".format(k)
            msg = msg + ")"
            raise NotImplementedError(msg)

        if 'override_{}'.format(robot_name) in conf:
            override_params = conf['override_{}'.format(robot_name)]
            conf['agent'].update(override_params.get('agent', {}))
            conf['agent']['env'][1].update(override_params.get('env_params', {}))
            conf['policy'].update(override_params.get('policy', {}))

        if 'imax' not in conf['agent']:
            conf['agent']['imax'] = 5

        if 'RESULT_DIR' in os.environ:
            exp_name = conf['agent']['data_save_dir'].split('/')[-1]
            conf['agent']['data_save_dir'] = '{}/{}'.format(os.environ['RESULT_DIR'], exp_name)

        self._hyperparams = conf
        self.agentparams, self.policyparams, self.envparams = conf['agent'], conf['policy'], conf['agent']['env'][1]

        self.envparams['robot_name'] = self.agentparams['robot_name'] = robot_name
        self._is_bench = is_bench
        if is_bench:
            self.task_mode = '{}/exp'.format(robot_name)
            self.agentparams['env'][1]['start_at_neutral'] = True     # robot should start at neutral during benchmarks
        else:
            self.task_mode = '{}/{}'.format(robot_name, conf.get('mode', 'train'))

        if 'register_gtruth' in self.policyparams:
            assert 'register_gtruth' not in self.agentparams, "SHOULD BE IN POLICY PARAMS"
            self.agentparams['register_gtruth'] = self.policyparams['register_gtruth']
        self._ngpu = ngpu
        self._gpu_id = gpu_id

        #since the agent interacts with Sawyer, agent creation handles recorder/controller setup
        self.agent = self.agentparams['type'](self.agentparams)
        self.policy = self.policyparams['type'](self.agentparams, self.policyparams, self._gpu_id, self._ngpu)

        robot_dir = self.agentparams['data_save_dir'] + '/{}'.format(robot_name)
        if not os.path.exists(robot_dir):
            os.makedirs(robot_dir)

        self._ck_path = self.agentparams['data_save_dir'] + '/{}/checkpoint.pkl'.format(robot_name)
        self._ck_dict = {'ntraj': 0, 'broken_traj': []}
        if resume:
            if resume == -1 and os.path.exists(self._ck_path):
                with open(self._ck_path, 'rb') as f:
                    self._ck_dict = pkl.load(f)
            else:
                self._ck_dict['ntraj'] = max(int(resume), 0)

        self._hyperparams['start_index'] = self._ck_dict['ntraj']

    def run(self):
        if not self._is_bench:
            for i in xrange(self._hyperparams['start_index'], self._hyperparams['end_index']):
                self.take_sample(i)
        else:
            itr = 0
            continue_collection = True
            while continue_collection:
                self.take_sample(itr)
                itr += 1
                continue_collection = 'y' in raw_input('Continue collection? (y if yes):')
        self.agent.cleanup()

    def _get_bench_name(self):
        name = raw_input('input benchmark name: ')
        while len(name) < 2:
            print('please choose a name > 2 characters long')
            name = raw_input('input benchmark name: ')
        return name

    def take_sample(self, sample_index):
        data_save_dir = self.agentparams['data_save_dir'] + '/' + self.task_mode

        if self._is_bench:
            bench_name = self._get_bench_name()
            traj_folder = '{}/{}'.format(data_save_dir, bench_name)
            self.agentparams['_bench_save'] = '{}/exp_data'.format(traj_folder)  # probably should develop a better way
            self.agentparams['benchmark_exp'] = bench_name                       # to pass benchmark info to agent
            self.agentparams['record'] = traj_folder + '/traj_data/record'
            print("Conducting experiment: {}".format(bench_name))
            traj_folder = traj_folder + '/traj_data'
            if os.path.exists(traj_folder):
                shutil.rmtree(traj_folder)
            os.makedirs(traj_folder)
        else:
            ngroup = self._hyperparams['ngroup']
            igrp = sample_index // ngroup
            group_folder = data_save_dir + '/traj_group{}'.format(igrp)
            if not os.path.exists(group_folder):
                os.makedirs(group_folder)

            traj_folder = group_folder + '/traj{}'.format(sample_index)
            print("Collecting sample {}".format(sample_index))

        agent_data, obs_dict, policy_out = self.agent.sample(self.policy, sample_index)

        if self._hyperparams['save_data']:
            self._save_raw_images(traj_folder, agent_data, obs_dict, policy_out)

        self._ck_dict['ntraj'] += 1
        ck_file = open(self._ck_path, 'wb')
        pkl.dump(self._ck_dict, ck_file)
        ck_file.close()

        print("CHECKPOINTED")

    def _save_raw_images(self, traj_folder, agent_data, obs_dict, policy_outputs):
        if not self._is_bench:
            if os.path.exists(traj_folder):
                shutil.rmtree(traj_folder)
            os.makedirs(traj_folder)

        if 'images' in obs_dict:
            images = obs_dict.pop('images')
            T, n_cams = images.shape[:2]
            for i in range(n_cams):
                os.mkdir(traj_folder + '/images{}'.format(i))
            for t in range(T):
                for i in range(n_cams):
                    cv2.imwrite('{}/images{}/im_{}.jpg'.format(traj_folder, i, t), images[t, i, :, :, ::-1])
        if 'goal_image' in obs_dict:
            goal_images = obs_dict.pop('goal_image')
            for n in range(goal_images.shape[0]):
                cv2.imwrite('{}/goal_image{}.jpg'.format(traj_folder, n),
                            (goal_images[n, :, :, ::-1] * 255).astype(np.uint8))

        with open('{}/agent_data.pkl'.format(traj_folder), 'wb') as file:
            pkl.dump(agent_data, file)
        with open('{}/obs_dict.pkl'.format(traj_folder), 'wb') as file:
            pkl.dump(obs_dict, file)
        with open('{}/policy_out.pkl'.format(traj_folder), 'wb') as file:
            pkl.dump(policy_outputs, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('robot_name', type=str, help="name of robot we're running on")
    parser.add_argument('experiment', type=str, help='experiment name')
    parser.add_argument('-r', nargs='?', dest='resume', const=-1,
                        default=False, help='Set flag if resuming training (-r if from checkpoint, -r <traj_num> otherwise')
    parser.add_argument('--gpu_id', type=int, default=0, help='value to set for cuda visible devices variable')
    parser.add_argument('--ngpu', type=int, default=1, help='number of gpus to use')
    parser.add_argument('--benchmark', action='store_true', default=False,
                        help='Add flag if this experiment is a benchmark')
    args = parser.parse_args()

    hyperparams = imp.load_source('hyperparams', args.experiment)
    conf = hyperparams.config

    env = RobotEnvironment(args.robot_name, conf, args.resume, args.ngpu, args.gpu_id, args.benchmark)
    env.run()
