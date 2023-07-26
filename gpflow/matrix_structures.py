# Copyright 2021 ST John
# Copyright 2016 James Hensman
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
from gpflow.config import default_float
from functools import reduce
import numpy as np

def BlockDiagMat(A, B):
    
    tl_shape = tf.stack([A.shape[0], B.shape[1]])
    br_shape = tf.stack([B.shape[0], A.shape[1]])
    top = tf.concat([A, tf.zeros(tl_shape, default_float())], axis=1)
    bottom = tf.concat([tf.zeros(br_shape, default_float()), B], axis=1)
    
    return tf.concat([top, bottom], axis=0)

def Rank1Mat(d, v):

    V = tf.expand_dims(v, 1)
    return tf.linalg.diag(d) + tf.matmul(V, V, transpose_b=True)

