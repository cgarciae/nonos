import tensorflow as tf
from builder import *
import nn

# Monkey Patch TensorFlow
tf.python.framework.ops.Tensor.builder = builder
