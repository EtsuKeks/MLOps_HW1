import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from swish import Swish
from torch import Tensor
from torch import nn
import tensorflow as tf
import numpy
import time

def test_timings(func, test_arrays):
    start_time = time.time()
    for test_array in test_arrays:
        _ = func(test_array)
    end_time = time.time()
    return round(end_time - start_time, 5)

def compare(test_arrays):
    print(
        "Swish (Pure C++): {0} seconds".format(
            test_timings(Swish, map(lambda arr: arr.tolist(), test_arrays))
        )
    )
    print(
        "Swish (Tensorflow): {0} seconds".format(
            test_timings(tf.keras.activations.silu, test_arrays)
        )
    )
    print(
        "Swish (PyTorch): {0} seconds".format(
            test_timings(nn.SiLU, map(lambda arr: Tensor(arr), test_arrays))
        )
    )

test_arrays = []
for _ in range(100):
    test_arrays.append(numpy.random.rand(500000))

compare(test_arrays)