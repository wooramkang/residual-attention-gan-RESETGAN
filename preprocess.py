from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
sys.path.append(os.path.abspath((os.path.dirname(__file__)))+'/preprocessing')
print(os.path.abspath((os.path.dirname(__file__)))+'/preprocessing')


def Make_embedding(x_train=None, x_test=None):
    if x_train is not None:
        x_train = x_train.astype('float32') / 255

    if x_test is not None:
        x_test = x_test.astype('float32') / 255

    return x_train, x_test

