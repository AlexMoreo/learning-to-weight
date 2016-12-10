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

    def __init__(self, plotmode, plotdir, target_function, net_idf_wrapper, plotpoints):
        self.target_function = target_function
        self.net_idf_wrapper = net_idf_wrapper
        self.plotdir = plotdir
        create_if_not_exists(plotdir)
        self.plotext = None
        self.mode = plotmode
        self.plotcount = 0
        self.plotpoints = plotpoints


    def plot_coordinates(self, div=50, x1_range=(0.0, 1.0), x2_range=(0.0, 1.0)):
        x1_div = np.linspace(x1_range[0], x1_range[1], div)
        x2_div = np.linspace(x2_range[0], x2_range[1], div)
        points = itertools.product(x1_div, x2_div)
        return zip(*[p for p in points])

    def _get_plot_points(self, x1_range, x2_range, div=40):
        x1_, x2_ = self.plot_coordinates(div=div, x1_range=x1_range, x2_range=x2_range)
        y_ = []
        for xi_ in zip(x1_, x2_):
            y_.append(self.net_idf_wrapper(xi_))
        y_ = np.reshape(y_, len(x1_))
        return x1_, x2_, y_

    def _get_target_points(self, x1_range, x2_range, div=40):
        x1_, x2_ = self.plot_coordinates(div=div, x1_range=x1_range, x2_range=x2_range)
        y_ = [self.target_function(x1_[i], x2_[i]) for i in range(len(x1_))]
        return x1_, x2_, y_

    def plot(self, step):
        if self.mode == 'off': return

        def _set_figure_labels(ax, title):
            ax.set_title(title)
            ax.set_xlabel('tpr')
            ax.set_ylabel('fpr')

        fig_pos = 1
        n_figs = 2 if self.target_function is None else 3
        fig = plt.figure(figsize=plt.figaspect(.8 / n_figs))

        # plot the target function
        if n_figs == 3:
            ax = fig.add_subplot(1, n_figs, fig_pos, projection='3d')
            _set_figure_labels(ax, 'Target function')
            x1, x2, y = self._get_target_points(x1_range=[0, 1], x2_range=[0, 1])
            ax.plot_trisurf(x1, x2, y, linewidth=0.2, cmap=cm.jet)
            fig_pos += 1

        # plot the learned function in ranges 0-1
        ax = fig.add_subplot(1, n_figs, fig_pos, projection='3d')
        _set_figure_labels(ax, 'Learnt function' + (' (%d steps)' % step if step else ''))
        x1, x2, y_ = self._get_plot_points(x1_range=[0, 1], x2_range=[0, 1])
        ax.plot_trisurf(x1, x2, y_, linewidth=0.2, cmap=cm.jet)
        fig_pos += 1

        ax = fig.add_subplot(1, n_figs, fig_pos, projection='3d')
        _set_figure_labels(ax, 'Learnt function' + (' (%d steps, Zoom)' % step if step else ''))
        tpr_points, fpr_points = zip(*self.plotpoints)
        x1_range = [min(tpr_points), max(tpr_points)]
        x2_range = [min(fpr_points), max(fpr_points)]
        x1, x2, y_ = self._get_plot_points(x1_range=x1_range, x2_range=x2_range)
        ax.plot_trisurf(x1, x2, y_, linewidth=0.2, cmap=cm.jet, alpha=0.6)
        min_y, max_y = min(y_), max(y_)
        ax.scatter(tpr_points, fpr_points, zs=min_y - 0.2 * (max_y - min_y), s=10, c='b', marker='o')
        ax.set_xlim3d(x1_range)
        ax.set_ylim3d(x2_range)
        fig_pos += 1

        if self.mode == 'show':
            plt.show()
        elif self.mode == 'img':
            plotpath = os.path.join(self.plotdir, 'IDF_step%.5d.pdf' % step)
            plt.savefig(plotpath)
        elif self.mode == 'vid':
            plotpath = os.path.join(self.plotdir, 'IDF_VID%.5d.png' % self.plotcount)
            plt.savefig(plotpath)
        self.plotcount += 1

