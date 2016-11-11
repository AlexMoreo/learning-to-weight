from __future__ import division

import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from helpers import *
import numpy as np
import itertools
from scipy.stats import norm

x_size = 2


def plot_coordinates(div=50):
    x1_div = np.linspace(0.0, 1.0, div)
    x2_div = np.linspace(0.0, 1.0, div)
    points = itertools.product(x1_div, x2_div)
    return zip(*[p for p in points])


def comp_plot(x1, x2, y_):
    y = [target_function([x1[i], x2[i]]) for i in range(len(x1))]
    fig = plt.figure(figsize=plt.figaspect(0.4))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.set_title('Target function')
    ax.plot_trisurf(x1, x2, y, linewidth=0.2, cmap=cm.jet)
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.set_title('Learnt function')
    ax.plot_trisurf(x1, x2, y_, linewidth=0.2, cmap=cm.jet)
    plt.show()


def target_function(x):
    assert (len(x) == x_size)
    # return 0.1+(math.cos(x[0]*10)+math.sin(x[1]*10))/2.0
    # a=x[0]*2-1
    # b=x[1]*2-1
    # return math.sin(math.pi*a*b)/0.5+a**2+math.e**a
    gauss1 = norm.pdf((x[0] - 0.5) * 6) * norm.pdf((x[1] - 0.5) * 7) * 2
    gauss2 = norm.pdf((x[0] - 0.7) * 8) * norm.pdf((x[1] - 0.3) * 11)
    return max(gauss1, gauss2)


def sample():
    x = [random.random() for _ in range(x_size)]
    y = target_function(x)
    return (x, y)


def batch(batch_size=1):
    next = [sample() for _ in range(batch_size)]
    return zip(*next)


def scale_p(xi, s):
    new_p = xi + s * random.random()
    return min(max(0.0, new_p), 1.0)


def batch_local(batch_size=100):
    scale = 0.001
    x, _ = sample()
    # xs = [(scale_p(x[0],scale), scale_p(x[1],scale)) for _ in range(batch_size)]
    xs = [[sum(x) for x in zip(x, offset)] for x, offset in zip([x for x in range(batch_size)], [
        [(random.random() - 0.5) * scale for _ in range(x_size)] for _ in range(batch_size)])]
    ys = [target_function(x) for x in xs]
    return xs, ys


def main(argv=None):
    graph = tf.Graph()
    with graph.as_default():
        # Placeholders
        x = tf.placeholder(tf.float32, shape=[None, x_size])
        y = tf.placeholder(tf.float32, shape=[None])
        # keep_p = tf.placeholder(tf.float32, name='dropoupt_keep_p')

        ff_out = ff_multilayer(x, hidden_sizes=[100, 50]*3, non_linear_function=tf.nn.relu)
        _y = add_linear_layer(ff_out, 1, name='output_layer')

        loss = tf.reduce_mean(tf.square(tf.sub(y, _y)))
        # loss = tf.nn.l2_loss(y - _y)

        op_step = tf.Variable(0, trainable=False)
        rate = tf.train.exponential_decay(.01, op_step, 1, 0.99999, staircase=True)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=rate).minimize(loss, global_step=op_step)

    # ---------------------------------------------------------------------
    # Graph run
    # ---------------------------------------------------------------------

    show_step = 1000
    valid_step = show_step * 10
    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        l_ave = 0.0
        for step in range(1, 1000000):
            x_, y_ = batch()
            _, l, lr = session.run([optimizer, loss, rate], feed_dict={x: x_, y: y_})
            l_ave += l

            if step % show_step == 0:
                print('[step=%d][lr=%.10f] loss=%.7f' % (step, lr, l_ave / show_step))
                l_ave = 0.0

            if step % valid_step == 0 and FLAGS.plot:
                x1_, x2_ = plot_coordinates(div=40)
                y_ = _y.eval(feed_dict={x: zip(x1_, x2_)})
                y_ = np.reshape(y_, len(x1_))
                comp_plot(x1_, x2_, y_)


# -------------------------------------
if __name__ == '__main__':
    flags = tf.app.flags
    FLAGS = flags.FLAGS

    flags.DEFINE_boolean('plot', True, 'Plots the current function model.')

    # plot_function()
    tf.app.run()
