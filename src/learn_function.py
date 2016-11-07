from __future__ import division

import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from helpers import *
import numpy as np
import itertools

x_size=2


def plot_coordinates(div=50):
    x1_div = np.linspace(0.0, 1.0, div)
    x2_div = np.linspace(0.0, 1.0, div)
    points = itertools.product(x1_div, x2_div)
    return zip(*[p for p in points])

def plot_function():
    x1,x2 = plot_coordinates()
    y = [target_function([x1[i],x2[i]]) for i in range(len(x1)) ]
    plot_points(x1,x2,y)

def plot_points(x1,x2,y):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(x1, x2, y, linewidth=0.2)
    plt.show()

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
    assert(len(x)==x_size)
    return 0.2+(x[0]**2+x[1])/2.0

def sample():
    x = [random.random() for _ in range(x_size)]
    y = target_function(x)
    return (x,y)

def batch(batch_size = 100):
    next = [sample() for _ in range(batch_size)]
    return zip(*next)

def main(argv=None):
    hidden_size = 30

    graph = tf.Graph()
    with graph.as_default():
      # Placeholders
      x = tf.placeholder(tf.float32, shape=[None,x_size])
      y = tf.placeholder(tf.float32, shape=[None])

      # Model parameters
      w1, b1 = projection_weights(x_size, hidden_size, 'x-hidden')
      w2, b2 = projection_weights(hidden_size, hidden_size, 'hidden-hidden')
      w3, b3 = projection_weights(hidden_size, hidden_size, 'hidden-hidden')
      w4, b4 = projection_weights(hidden_size, 1, 'hidden-y')

      h1 = tf.nn.relu(tf.matmul(x,w1)+b1)
      h2 = tf.nn.relu(tf.matmul(h1,w2)+b2)
      h3 = tf.nn.relu(tf.matmul(h2, w3) + b3)
      _y = (tf.matmul(h3,w4)+b4)
      #_y = tf.nn.sigmoid(tf.matmul(x, w1)+b1)

      #reg = (tf.nn.l2_loss(w1)+tf.nn.l2_loss(b1))/hidden_size
      def uni_reg(w):
          cov = tf.matmul(tf.transpose(w),w)
          cov = cov - tf.diag(cov)
          return tf.nn.l2_loss(cov)/w.get_shape().as_list()[1]

      r = 0.0*(uni_reg(w1)+uni_reg(w2)+uni_reg(w3))

      #loss = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(y, _y)))) #+ reg
      loss = tf.reduce_mean(tf.abs(tf.sub(y, _y))) #+ r

      op_step = tf.Variable(0, trainable=False)
      rate = tf.train.exponential_decay(.0001, op_step, 1, 0.9999)
      optimizer = tf.train.GradientDescentOptimizer(learning_rate=rate).minimize(loss, global_step=op_step)


    # ---------------------------------------------------------------------
    # Graph run
    # ---------------------------------------------------------------------

    show_step = 100
    valid_step = 10000
    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        l_ave=0.0
        for step in range(1,1000000):
            x_, y_ = batch()
            _,l,lr = session.run([optimizer, loss, rate], feed_dict={x:x_, y:y_})
            l_ave += l

            if step % show_step == 0:
                print('[step=%d][lr=%.5f] loss=%.7f [r=%.7f]' % (step, lr, l_ave / show_step, -1))
                l_ave = 0.0

            if step % valid_step == 0 and FLAGS.plot:
                x1_, x2_ = plot_coordinates(div=10)
                y_ = _y.eval(feed_dict={x: zip(x1_, x2_)})
                y_ = np.reshape(y_,len(x1_))
                comp_plot(x1_,x2_,y_)



#-------------------------------------
if __name__ == '__main__':
    flags = tf.app.flags
    FLAGS = flags.FLAGS

    flags.DEFINE_boolean('plot', True, 'Plots the current function model.')

    #plot_function()
    tf.app.run()

