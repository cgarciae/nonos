import tensorflow as tf
import numpy as np


# Decorators
def immutable(func):
    """Method decorator. Passes a copy of the builder the method so that the original object remains un touched."""
    def func_wrapper(self, *args, **kwargs):
        cp = self.copy()
        return func(cp, *args, **kwargs)
    return func_wrapper



class Builder(object):
    """docstring for Builder"""
    def __init__(self):
        super(Builder, self).__init__()
        self.tensor = None
        self.variables = {}

    def copy(self):
        cp = Builder()
        cp.tensor = self.tensor
        cp.variables = self.variables.copy()

        return cp

    @immutable
    def connect_weights(self, size, weights_name="w"):
        m = int(self.tensor.get_shape()[1])
        n = size

        w = tf.Variable(tf.random_uniform([m, n], -1.0, 1.0), name=weights_name)

        self.variables[w.name] = w
        self.tensor = tf.matmul(self.tensor, w)

        return self

    @immutable
    def connect_bias(self, bias_name="b"):
        m = int(self.tensor.get_shape()[1])

        b = tf.Variable(tf.random_uniform([m], -1.0, 1.0), name=bias_name)

        self.variables[b.name] = b
        self.tensor += b

        return self


    @immutable
    def connect_layer(self, size, fn=None, name=None, weights_name=None, bias=True, bias_name=None):

        self = self.connect_weights(size, weights_name=weights_name)

        if bias:
            self = self.connect_bias(bias_name=bias_name)

        if fn:
            self.tensor = fn(self.tensor, name=name)

        return self

    @immutable
    def map(self, fn, *args, **kwargs):
        self.tensor = fn(self.tensor, *args, **kwargs)
        return self

    @immutable
    def then(self, fn):
        return fn(self)

    @immutable
    def branch(self, fn):
        return fn(self)

    @immutable
    def branch_reduce(self, fn):
        branches = fn(self)
        return add_branches(branches)


## Module Funs
def builder(tensor):
    builder = Builder()
    builder.tensor = tensor

    return builder

def add_branches(branches):
    tensor = None
    variables = {}

    for builder in branches:
        if tensor == None:
            tensor = builder.tensor
        else:
            tensor += builder.tensor

        variables.update(builder.variables)

    builder = Builder()
    builder.tensor = tensor
    builder.variables = variables

    return builder
