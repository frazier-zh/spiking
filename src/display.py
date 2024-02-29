### 2019 Aug 21
### Zihang
### ATGroup, NUS

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
import numpy as np

def weight_to_image(array, w_min=0, w_max=1, fname=None, size=None, type_mask=None):
    num = array.shape[0]
    cols = int(np.ceil(np.sqrt(num)))
    rows = int((num-1)/cols)+1

    if size is None:
        pixels = array.shape[1]
        y = int(np.ceil(np.sqrt(pixels)))
        x = int((pixels-1)/y)+1
        size = (x,y)

    plt.clf()
    fig = plt.figure(figsize=(2*cols, 2*rows))
    plt.subplots_adjust(bottom=.01, left=.01, right=.99, top=.90)

    for i in range(0, num):
        if size is not None:
            data = array[i].reshape(size)
            if type_mask is not None:
                type = type_mask[i].reshape(size)
        else:
            data = array[i]

        axs = fig.add_subplot(rows, cols, i+1)
        axs.set_xticks([])
        axs.set_yticks([])

        img = axs.imshow(data, cmap='gray', vmin=w_min, vmax=w_max)

        # mark for anti-stdp connection
        if type_mask is not None:
            for x in range(size[0]):
                for y in range(size[1]):
                    if type[x][y]:
                        rect = patches.Rectangle((x-0.48, y-0.48), 0.96, 0.96, linewidth=1, facecolor='none', edgecolor='r')
                        axs.add_patch(rect)

    # colorbar
    fig.colorbar(img, cax=fig.add_axes([0.2, 0.93, 0.6, 0.03]), orientation='horizontal')

    if fname is not None:
        plt.savefig(fname+'.png')
    else:
        plt.show()

    plt.close()
    
def neuron_to_image(array1, array2, ymax=1.5, fname=None):
    num = array1.shape[0]
    cols = int(np.ceil(np.sqrt(num)))
    rows = int((num-1)/cols+1)

    plt.clf()
    fig = plt.figure(figsize=(4*cols, 4*rows))
    plt.subplots_adjust(bottom=.1, left=.1, right=.99, top=.99)

    tick = range(len(array1[0]))

    for i in range(0, num):
        ax = fig.add_subplot(rows, cols, i+1)
        ax.plot(tick, array1[i])
        ax.plot(tick, array2[i])
        ax.set_ylim(-0.1, ymax)

    if fname is not None:
        plt.savefig(fname+'.png')
    else:
        plt.show()  
    plt.close()

import sys
def progress(count, total, status='', bar_len = 50):
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '>' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write(f'[{bar}] {percents:.2f}% ({count} / {total}) {status}\r')
    sys.stdout.flush()