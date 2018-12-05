from sim.util.synchronize_tfrecs import sync
from multiprocessing import Pool, Process, Manager
import sys
import argparse
import importlib.machinery
import importlib.util
from sim.simulator import Sim
from sim.benchmarks import perform_benchmark
import copy
import random
import numpy as np
from visual_mpc.agent.utils.traj_saver import record_worker
import re
import os
from sim.util.combine_score import combine_scores
import ray
import pdb


class SynchCounter:
    def __init__(self, manager):
        self._lock, self._value = manager.Lock(), manager.Value('i', 0)

    def ret_increment(self):
        with self._lock:
            ret_val = self._value.value
            self._value.value += 1
        return ret_val

    @property
    def value(self):
        with self._lock:
            ret_val = self._value.value
        return ret_val


def worker(conf, iex=-1, ngpu=1):
    print('started process with PID:', os.getpid())
    print('making trajectories {0} to {1}'.format(
        conf['start_index'],
        conf['end_index'],
    ))

    random.seed(None)
    np.random.seed(None)

    s = Sim(conf)
    s.run()

def bench_worker(conf, iex=-1, ngpu=1):
    print('started process with PID:', os.getpid())
    random.seed(None)
    np.random.seed(None)
    perform_benchmark(conf, iex, gpu_id=conf['gpu_id'], ngpu=ngpu)


def check_and_pop(dict, key):
    if dict.pop(key, None) is not None:
        print('popping key: {}'.format(key))


def main():
    parser = argparse.ArgumentParser(description='run parllel data collection')
    parser.add_argument('experiment', type=str, help='experiment name')
    parser.add_argument('--nworkers', type=int, help='use multiple threads or not', default=1)
    parser.add_argument('--gpu_id', type=int, help='the starting gpu_id', default=0)
    parser.add_argument('--ngpu', type=int, help='the number of gpus to use', default=1)

    parser.add_argument('--nsplit', type=int, help='number of splits', default=-1)
    parser.add_argument('--isplit', type=int, help='split id', default=-1)
    parser.add_argument('--cloud', dest='cloud', action='store_true', default=False)
    parser.add_argument('--benchmark', dest='do_benchmark', action='store_true', default=False)

    parser.add_argument('--iex', type=int, help='if different from -1 use only do example', default=-1)

    args = parser.parse_args()
    hyperparams_file = args.experiment
    gpu_id = args.gpu_id

    n_worker = args.nworkers
    if args.nworkers == 1:
        parallel = False
    else:
        parallel = True
    print('parallel ', bool(parallel))

    loader = importlib.machinery.SourceFileLoader('mod_hyper', hyperparams_file)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    hyperparams = mod.config

    if args.nsplit != -1:
        assert args.isplit >= 0 and args.isplit < args.nsplit, "isplit should be in [0, nsplit-1]"
       
        n_persplit = max((hyperparams['end_index'] + 1 - hyperparams['start_index']) / args.nsplit, 1)
        hyperparams['end_index'] = int((args.isplit + 1) * n_persplit + hyperparams['start_index'] - 1)
        hyperparams['start_index'] = int(args.isplit * n_persplit + hyperparams['start_index'])

    n_traj = hyperparams['end_index'] - hyperparams['start_index'] + 1
    traj_per_worker = int(n_traj // np.float32(n_worker))
    start_idx = [hyperparams['start_index'] + traj_per_worker * i for i in range(n_worker)]
    end_idx = [hyperparams['start_index'] + traj_per_worker * (i+1)-1 for i in range(n_worker)]


    if 'gen_xml' in hyperparams['agent']: #remove old auto-generated xml files
        try:
            os.system("rm {}".format('/'.join(str.split(hyperparams['agent']['filename'], '/')[:-1]) + '/auto_gen/*'))
        except: pass

    if args.do_benchmark:
        use_worker = bench_worker
    else: use_worker = worker

    if 'RESULT_DIR' in os.environ:
        if 'exp_name' in hyperparams:
            exp_name = hyperparams['exp_name']
        elif 'record' in hyperparams['agent']:
            exp_name = [f for f in hyperparams['agent']['record'].split('/') if f != 'record' and len(f) > 0][-1]
        elif 'data_save_dir' in hyperparams['agent']:
            exp_name = hyperparams['agent']['data_save_dir'].split('/')[-1]
        else:
            raise NotImplementedError("can't find exp name")
        result_dir = '{}/{}'.format(os.environ['RESULT_DIR'], exp_name)

        if 'verbose' in hyperparams['policy'] and not os.path.exists(result_dir + '/verbose'):
            os.makedirs(result_dir + '/verbose')

        if 'data_save_dir' in hyperparams['agent']:
            hyperparams['agent']['data_save_dir'] = result_dir

    elif 'EXPERIMENT_DIR' in os.environ:
        subpath = hyperparams['current_dir'].partition('experiments')[2]
        result_dir = os.path.join(os.environ['EXPERIMENT_DIR'] + subpath)
    elif args.cloud:
        check_and_pop(hyperparams, 'save_raw_images')
        check_and_pop(hyperparams['agent'], 'make_final_gif')
        check_and_pop(hyperparams['agent'], 'make_final_gif_pointoverlay')
        hyperparams['agent']['data_save_dir'] = '/result/'    # by default save code to the /result folder in docker image
    else:
        result_dir = hyperparams['current_dir'] + '/verbose'

    if 'master_datadir' in hyperparams['agent']:
        ray.init()
        sync_todo_id = sync.remote(hyperparams['agent'])
        print('launched sync')

    if 'data_save_dir' in hyperparams['agent']:
        record_queue, record_saver_proc, counter = prepare_saver(hyperparams)

    if args.iex != -1:
        hyperparams['agent']['iex'] = args.iex

    conflist = []
    for i in range(n_worker):
        modconf = copy.deepcopy(hyperparams)
        modconf['start_index'] = start_idx[i]
        modconf['end_index'] = end_idx[i]
        modconf['ntraj'] = n_traj
        modconf['gpu_id'] = i + gpu_id
        modconf['result_dir'] = result_dir
        if 'data_save_dir' in hyperparams['agent']:
            modconf['record_saver'] = record_queue
            modconf['counter'] = counter
        conflist.append(modconf)
    if parallel:
        p = Pool(n_worker)
        p.map(use_worker, conflist)
    else:
        use_worker(conflist[0], args.iex, args.ngpu)

    if 'data_save_dir' in hyperparams['agent'] and not hyperparams.get('save_raw_images', False):
        record_queue.put(None)           # send flag to background thread that it can end saving after it's done
        record_saver_proc.join()         # joins thread and continues execution

    if 'master_datadir' in hyperparams['agent']:
        ray.wait([sync_todo_id])

    if args.do_benchmark:
        pdb.set_trace()
        combine_scores(hyperparams, result_dir)
        sys.exit()


def prepare_saver(hyperparams):
    m = Manager()
    record_queue, synch_counter = m.Queue(), SynchCounter(m)
    save_dir, T = hyperparams['agent']['data_save_dir'] + '/records', hyperparams['agent']['T']
    if hyperparams.get('save_data', True) and not hyperparams.get('save_raw_images', False):
        seperate_good, traj_per_file = hyperparams.get('seperate_good', False), hyperparams.get('traj_per_file', 16)
        record_saver_proc = Process(target=record_worker, args=(
        record_queue, save_dir, T, seperate_good, traj_per_file, hyperparams['start_index']))
        record_saver_proc.start()
    else:
        record_saver_proc = None
    return record_queue, record_saver_proc, synch_counter


def sorted_alphanumeric(l):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)


if __name__ == '__main__':
    main()
