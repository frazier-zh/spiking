from snn.base import preceptron_lateral_layer

from data_generator import gconv_encode, gen_base
import os
import time
from datetime import datetime, timedelta
import pickle
import numpy as np

import sys
def progress(count, total, status='', bar_len = 50):
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '>' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write(f'[{bar}] {percents:.2f}% ({count} / {total}) {status}\r')
    sys.stdout.flush()

import matplotlib.pyplot as plt
def visualize(array, fname, w_min=0, w_max=1, size=None, arange=None):
    num = array.shape[0]

    if arange is None:
        cols = int(np.ceil(np.sqrt(num)))
        rows = int((num-1)/cols)+1
    else:
        cols, rows = arange

    if size is None:
        if len(array.shape) == 2:
            pixels = array.shape[1]
            y = int(np.ceil(np.sqrt(pixels)))
            x = int((pixels-1)/y)+1
            size = [x, y]
        elif len(array.shape) == 3:
            size = array.shape[1:3]

    scale = np.max(size)/10

    fig = plt.figure(figsize=(scale*cols, scale*rows))
    plt.subplots_adjust(bottom=.01, left=.01, right=.99, top=.99)

    for i in range(0, num):
        data = array[i].reshape(size)

        axs = fig.add_subplot(rows, cols, i+1)
        axs.set_xticks([])
        axs.set_yticks([])

        img = axs.imshow(data, vmin=w_min, vmax=w_max, cmap='gray')
    # colorbar
    #fig.colorbar(img, cax=fig.add_axes([0.2, 0.93, 0.6, 0.03]), orientation='horizontal')

    plt.savefig(fname+'.png')
    plt.close()

class NeuralNet:
    def __init__(self, path):
        if not os.path.isdir(path):
            os.mkdir(path)
        self.output_path = path

        self.net = preceptron_lateral_layer(25, 25)
        self.step = 0

    def _single(self):
        # for profile purposes
        for img, label in gen_base('mnist-train'):
            for encoded_img in gconv_encode(img, [28, 28], [5, 5]):
                self.net.forward(encoded_img)
            break

    def train(self, train_steps, batch_size=1000):
        step = 0
        total_elapsed_time = 0
        
        progress(step, train_steps, status=f'Time of completion: ...')
        self.synapse_savefig()
        for img, label in gen_base('mnist-train'):
            # timing
            step_start_time = time.time()

            # training
            for encoded_img in gconv_encode(img, [28, 28], [5, 5]):
                self.net.forward(encoded_img)

            # timing
            step_time = time.time() - step_start_time
            total_elapsed_time += step_time
            avg_step_time = total_elapsed_time / (step + 1)
            remaining_time = (train_steps - step - 1) * avg_step_time
            completion_time = datetime.now() + timedelta(seconds=remaining_time)
            progress(step, train_steps, status=f'Time of completion: {completion_time.strftime("%d/%m %H:%M:%S")}')

            # counter
            self.step += 1
            step += 1
            
            # save state
            if step%batch_size==0:
                with open(self.output_path+f'net_{self.step}.pkl', 'wb') as f:
                    pickle.dump(self.net, f)
                self.synapse_savefig()

            if step>=train_steps:
                break

    def synapse_savefig(self):
        visualize(self.net.synapses, self.output_path+f'synapse_{self.step}_synapse', w_min=0, w_max=1, arange=[5, 5])

net = NeuralNet('./snapshots_synapse_new/')

train_steps = 6000
batch_size = 100

print(f"""\
======== SNN Simulator ========
output_path:    {net.output_path}
train_steps:    {train_steps}
batch_size:     {batch_size}
===============================
""")

net.train(train_steps, batch_size)

# import cProfile
# cProfile.run('net._single()', 'profile_report')
# import pstats
# from pstats import SortKey
# p = pstats.Stats('profile_report')
# p.strip_dirs().sort_stats(SortKey.TIME).print_stats(20)