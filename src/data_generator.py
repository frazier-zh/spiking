### Created: 2019 Aug 21
### Modified: 2024 Feb 21
### Author: Zihang
### Aff: ATGroup, NUS

# path to datasets
dataset = {
    'mnist-train': "mnist_data/train-images.npy",
    'mnist-test': "mnist_data/test-images.npy",
    '5x5-train': "data/5x5-train-images.npy",
    '5x5-test': "data/5x5-test-images.npy"
}

dataset_label = {
    'mnist-train': "mnist_data/train-labels.npy",
    'mnist-test': "mnist_data/test-labels.npy",
    '5x5-train': "data/5x5-train-labels.npy",
    '5x5-test': "data/5x5-test-labels.npy"
}

from random import randrange, uniform, randint
import numpy as np
from snn.receptive_field import rf
from snn.spike_train import encode

def load_mnist(path, kind='train'):
    import os
    import gzip
    import numpy as np

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels

def generator(images, labels, shuffle=True):
    num = len(labels)
    indices = np.arange(num)
    while True:
        if shuffle:
            np.random.shuffle(indices)
        for i in range(num):
            yield images[indices[i]], labels[indices[i]]


# image data generator, return value (image, label)
# mode='train' for training data, 'test' for test data
def gen_base(dataset_name, shuffle=True):
    # setup dataset
    if dataset_name in dataset:
        images = np.load(dataset[dataset_name])
        labels = np.load(dataset_label[dataset_name])
    else:
        raise RuntimeError("Invalid dataset name \'{}\'.".format(dataset_name))
    num = len(labels)
    indices = np.arange(num)

    while True:
        if shuffle:
            np.random.shuffle(indices)

        for i in range(num):
            yield images[indices[i]], labels[indices[i]]

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def gen_cifar(dataset_name, shuffle=True, channel=0):
    if dataset_name == 'test':
        raw = unpickle('cifar-10-batches-py/test_batch')
        images = raw[b'data'][:, 32*32*channel:32*32*(channel+1)]
        labels = raw[b'labels']
    elif dataset_name == 'train':
        images = []
        labels = []
        for i in range(5):
            raw = unpickle('cifar-10-batches-py/data_batch_%d'%(i+1))
            images.append(raw[b'data'])
            labels.append(raw[b'labels'])

        images = np.concatenate(images, axis=0)[:, 32*32*channel:32*32*(channel+1)]
        labels = np.concatenate(labels, axis=0)

    num = len(labels)
    indices = np.arange(num)
    while True:
        if shuffle:
            np.random.shuffle(indices)

        for i in range(num):
            yield images[indices[i]], labels[indices[i]]

def gconv_encode(image, image_size, kernel=[3,3], stride=1):
    kernel_length = kernel[0] * kernel[1]
    half_k = np.array(kernel)//2

    padded_img = np.zeros(image_size+half_k*2)
    padded_img[half_k[0]:half_k[0]+image_size[0], \
        half_k[1]:half_k[1]+image_size[1]] = image.reshape(image_size)

    encoded_img = encode(padded_img.flatten()).reshape(*padded_img.shape, -1)
    for ix in range(image_size[0]//stride):
        for iy in range(image_size[1]//stride):
            x = stride*ix
            y = stride*iy
            if not padded_img[x:x+kernel[0], y:y+kernel[1]].any():
                continue
            yield encoded_img[x:x+kernel[0], y:y+kernel[1]].reshape(kernel_length, -1)

if __name__=='__main__':
    pass