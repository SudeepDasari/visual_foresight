import os
from multiprocessing import Manager, Process
import cv2
import imageio as io
import numpy as np
import logging


def start_file_worker():
    m = Manager()
    file_queue = m.Queue()
    saver_proc = Process(target=_file_worker, args=(file_queue,))
    saver_proc.start()
    return file_queue


def _make_parent_if_needed(file_name):
    parent_dir = os.path.dirname(file_name)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)


def _file_worker(file_queue):
    logging.debug('started file saver with PID:', os.getpid())
    data = file_queue.get(True)
    prepend_path = './'
    while data is not None:
        dat_type = data[0]
        if dat_type == 'path':
            prepend_path = data[1]
            if not os.path.exists(prepend_path):
                os.makedirs(prepend_path)
        elif dat_type == 'txt_file':
            save_path = '{}/{}'.format(prepend_path, data[1])
            _make_parent_if_needed(save_path)
            with open(save_path, 'w') as f:
                f.write(data[2])
                f.write('\n')
        elif dat_type == 'mov':
            save_path = '{}/{}'.format(prepend_path, data[1])
            _make_parent_if_needed(save_path)
            fps, frames = 4, data[2]
            if len(data) == 4:
                fps = data[3]
            writer = io.get_writer(save_path, fps=fps)
            [writer.append_data(f.astype(np.uint8)) for f in frames]
            writer.close()
        elif dat_type == 'img':
            save_path = '{}/{}'.format(prepend_path, data[1])
            _make_parent_if_needed(save_path)
            cv2.imwrite(save_path, data[2][:, :, ::-1])

        data = file_queue.get(True)
    return