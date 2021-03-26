import tensorflow as tf
import numpy

from sigmoid_like import sigmoid_like

# define reference operation
def sigmoid_like_ref(x):
    from tensorflow.keras import backend as kb
    y = kb.clip(0.9 * x, -0.2, 0.2)
    return kb.clip(y + 0.1*x, -0.5, 0.5) + 0.5

#  test
x = 4 * tf.random.uniform((224, 224, 32))
with tf.GradientTape() as g:
    g.watch(x)
    y_test = sigmoid_like_ref(x)
    y_test *= y_test
g_ref = g.gradient(y_test, x)

with tf.GradientTape() as g:
    g.watch(x)
    y_ref = sigmoid_like(x)
    y_ref *= y_ref
g_test = g.gradient(y_ref, x)

y_err = tf.reduce_max(tf.abs(y_ref - y_test))
g_err = tf.reduce_max(tf.abs(g_ref - g_test))
print('Max forward pass error:', y_err)
print('Max backward pass error:', g_err)


# plot
if y_err > 1e-6:
    x = numpy.linspace(-4, 4, 5000, dtype=numpy.float32)
    y_test = sigmoid_like(x)
    y_ref = sigmoid_like_ref(x)
    import matplotlib.pyplot as plt
    plt.plot(x, y_ref, label="Reference")
    plt.plot(x, y_test, label="Test")
    plt.legend()
    plt.grid(True)
    plt.show()

