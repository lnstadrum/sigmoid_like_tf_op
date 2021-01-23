import tensorflow as tf
import numpy
import os

script_dir = os.path.dirname(os.path.realpath(__file__))
lib = tf.load_op_library(os.path.join(script_dir, 'build', 'libsigmoid.so'))

@tf.python.framework.ops.RegisterGradient("SigmoidLike")
def _sigmoid_like_backprop(op, grad):
    return [lib.SigmoidLikeGradient(input=op.inputs[0], gradient=grad)] 

def sigmoid_like(x):
    return lib.SigmoidLike(input=x)
