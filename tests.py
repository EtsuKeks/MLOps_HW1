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

def compare(test_arrays, torch_swish, tensorflow_swish):
    print(
        "Swish (Pure C++): {0} seconds".format(
            test_timings(Swish, map(lambda arr: arr.tolist(), test_arrays))
        )
    )
    print(
        "Swish (Tensorflow): {0} seconds".format(
            test_timings(tensorflow_swish, test_arrays)
        )
    )
    print(
        "Swish (PyTorch): {0} seconds".format(
            test_timings(torch_swish, map(lambda arr: Tensor(arr), test_arrays))
        )
    )

torch_swish = nn.SiLU
tensorflow_swish = tf.keras.activations.silu

test_arrays = []
for _ in range(100):
    test_arrays.append(numpy.random.rand(500000))

compare(test_arrays, torch_swish, tensorflow_swish)