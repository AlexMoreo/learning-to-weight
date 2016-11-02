import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from helpers import *
import numpy as np
import itertools

x_size=2


def plot_coordinates():
    x1_div = np.linspace(0.0, 1.0, 50)
    x2_div = np.linspace(0.0, 1.0, 50)
    points = itertools.product(x1_div, x2_div)
    x1, x2 = [], []
    for x1_, x2_ in points:
        x1.append(x1_)
        x2.append(x2_)
    return x1,x2

def plot_function():
    x1,x2 = plot_coordinates()
    y = [target_function([x1[i],x2[i]]) for i in range(len(x1)) ]
    plot_points(x1,x2,y)

def plot_points(x1,x2,y):
    if FLAGS.noplot:
        return
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(x1, x2, y, linewidth=0.2)
    plt.show()


def target_function(x):
    assert(len(x)==x_size)
    return x[0]*x[1]

plot_function()
#sys.exit()

def sample():
    x = [random.random() for _ in range(x_size)]
    y = target_function(x)
    return (x,y)

def batch(batch_size = 64):
    next = [sample() for _ in range(batch_size)]
    return zip(*next)

def main(argv=None):
    hidden_size = 10

    graph = tf.Graph()
    with graph.as_default():
      # Placeholders
      x = tf.placeholder(tf.float32, shape=[None,x_size])
      y = tf.placeholder(tf.float32, shape=[None])

      # Model parameters
      w1, b1 = projection_weights(x_size, hidden_size, 'x-hidden')
      #w2, b2 = projection_weights(hidden_size, hidden_size, 'hidden-hidden')
      w3, b3 = projection_weights(hidden_size, 1, 'hidden-y')

      h1 = tf.nn.sigmoid(tf.matmul(x,w1)+b1)
      #h2 = tf.nn.relu(tf.matmul(h1,w2)+b2)
      _y = tf.nn.sigmoid(tf.matmul(h1,w3)+b3)

      loss = tf.reduce_mean(tf.square(y - _y))

      optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)

      # ----------------------------------------------------------------------------------------------
      # Add ops to save and restore all the variables.
      saver = tf.train.Saver(max_to_keep=1)

    # ---------------------------------------------------------------------
    # Graph run
    # ---------------------------------------------------------------------

    show_step = 10
    valid_step = 2500
    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        l_ave=0.0
        for step in range(1,1000000):
            #x_, y_ = batch()
            #_,l = session.run([optimizer, loss], feed_dict={x:x_,y:y_})
            #l_ave += l

            x1_, x2_ = plot_coordinates()
            x_ = zip(x1_, x2_)
            y_ = [target_function([x1_[i], x2_[i]]) for i in range(len(x1_))]
            _, l = session.run([optimizer, loss], feed_dict={x: x_, y: y_})
            l_ave += l

            if step % show_step == 0:
                print('[step=%d] loss=%.4f' % (step, l_ave / show_step))
                l_ave = 0.0

            if step % valid_step == 0:
                x1_, x2_ = plot_coordinates()
                y_ = _y.eval(feed_dict={x: zip(x1_, x2_)})
                y_ = np.reshape(y_,len(x1_))
                plot_points(x1_, x2_, y_)


#-------------------------------------
if __name__ == '__main__':
    flags = tf.app.flags
    FLAGS = flags.FLAGS

    tf.app.run()

