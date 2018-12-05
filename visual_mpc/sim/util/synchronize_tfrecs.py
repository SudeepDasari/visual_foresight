"this program runs on ngc and syncs data with a local master machine"
import time
import os
import ray


@ray.remote
def sync(agentparams):
    master_datadir = agentparams['master_datadir']
    master = agentparams.get('master', 'deepthought')
    local_datadir = '/result'

    while True:
        print('transfer tfrecords to master')
        cmd = 'rsync -a --update {} {}:{}'.format(local_datadir + '/', master, master_datadir)
        print('executing: {}'.format(cmd))
        os.system(cmd)
        time.sleep(10)

if __name__ == '__main__':
    conf = {}
    conf['master_datadir'] = '/raid/ngc2/pushing_data/cartgripper/mj_multi_obj_push3_75step'
    sync(0, conf)