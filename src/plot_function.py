import numpy as np
import itertools
from helpers import *
import os
import matplotlib
if 'MATPLOTLIB_USE' in os.environ:
    matplotlib.use(os.environ['MATPLOTLIB_USE'])
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

class PlotIdf:

    def __init__(self, plotmode, plotdir, target_function, net_idf_wrapper):
        self.target_function = target_function
        self.net_idf_wrapper = net_idf_wrapper
        self.plotdir = plotdir
        create_if_not_exists(plotdir)
        self.plotext = None
        self.mode = plotmode
        self.plotcount = 0


    def plot_coordinates(self, div=50):
        x1_div = np.linspace(0.0, 1.0, div)
        x2_div = np.linspace(0.0, 1.0, div)
        points = itertools.product(x1_div, x2_div)
        return zip(*[p for p in points])

    def comp_plot(self, x1, x2, y_, step):
        y = [self.target_function(x1[i], x2[i]) for i in range(len(x1))]
        fig = plt.figure(figsize=plt.figaspect(0.4))
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.set_title('Target function')
        ax.set_xlabel('tpr')
        ax.set_ylabel('fpr')
        ax.plot_trisurf(x1, x2, y, linewidth=0.2, cmap=cm.jet)
        ax = fig.add_subplot(1, 2, 2, projection='3d')
        ax.set_title('Learnt function' + (' (%d steps)' % step if step else ''))
        ax.set_xlabel('tpr')
        ax.set_ylabel('fpr')
        ax.plot_trisurf(x1, x2, y_, linewidth=0.2, cmap=cm.jet)
        if self.mode == 'show':
            plt.show()
        elif self.mode == 'img':
            plotpath = os.path.join(self.plotdir, 'IDF_step%.5d.pdf' % step)
            plt.savefig(plotpath)
        elif self.mode == 'vid':
            plotpath = os.path.join(self.plotdir, 'IDF_VID%.5d.png' % self.plotcount)
            plt.savefig(plotpath)
        self.plotcount += 1

    def plot(self, step):
        if self.mode == 'off': return
        x1_, x2_ = self.plot_coordinates(div=40)
        y_ = []
        for xi_ in zip(x1_, x2_):
            #y_.append(idf_prediction.eval(feed_dict={x_func: [xi_], keep_p: 1.0}))
            y_.append(self.net_idf_wrapper(xi_))
        y_ = np.reshape(y_, len(x1_))
        self.comp_plot(x1_, x2_, y_, step=step)
