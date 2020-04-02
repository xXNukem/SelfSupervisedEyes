from __future__ import absolute_import
from __future__ import print_function
import numpy as np

import random
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout, Lambda, Conv2D,MaxPooling2D
from keras.optimizers import RMSprop
from keras import backend as K

num_classes = 10
epochs = 20



def create_pairs(x, digit_indices):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(num_classes)]) - 1
    for d in range(num_classes):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, num_classes)
            dn = (d + inc) % num_classes
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels)

#Red Base
def create_base_network(input_shape):
    input = Input(shape=input_shape)
    x = Flatten()(input)
    x = Conv2D(96, (11, 11), activation='relu', padding='same', name='conv1')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)
        #LRN1
    x = Conv2D(384, (3, 3), activation='relu', padding='same', name='conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(x)
        #LRN2
    x = Conv2D(384, (3, 3), activation='relu', padding='same', name='conv3')(x)
    x = Conv2D(384, (3, 3), activation='relu', padding='same', name='conv4')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv5')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv6')(x)

    x = Dense(4096, activation='relu', name='fc1')(x)
    return Model(input, x)


