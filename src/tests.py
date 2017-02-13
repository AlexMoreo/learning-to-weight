from src.utils.plot_function import *

if 'MATPLOTLIB_USE' in os.environ:
    matplotlib.use(os.environ['MATPLOTLIB_USE'])
import matplotlib.pyplot as plt
from matplotlib import cm


def plot_coordinates(xdiv=50, ydiv=50, x1_range=(0.0, 1.0), x2_range=(0.0, 1.0)):
    x1_div = np.linspace(x1_range[0], x1_range[1], xdiv)
    x2_div = np.linspace(x2_range[0], x2_range[1], ydiv)
    points = itertools.product(x1_div, x2_div)
    return zip(*[p for p in points])


def _get_target_points(x1_range, x2_range, xdiv=50, ydiv=50, func1=None, func2=None):
    x1_, x2_ = plot_coordinates(xdiv=xdiv, ydiv=ydiv, x1_range=x1_range, x2_range=x2_range)
    y_ = [func2(apply_tsr2(x1_[i], x2_[i], 1000, func1)) for i in range(len(x1_))]
    return x1_, x2_, y_

def identity(x):
    return x

def square(x):
    #return math.exp(x*3)
    return x**3

def lg(x):
    #return math.pow(x+.001,.3)
    return math.pow(x+0.0001, .333)
    #return math.log(x+0.001)

def plot():
    def _set_figure_labels(ax, title):
        ax.set_title(title)
        ax.set_xlabel('tpr')
        ax.set_ylabel('fpr')

    fig_pos = 1
    n_figs = 3
    fig = plt.figure(figsize=plt.figaspect(.8 / n_figs))

    # plot the target function
    ax = fig.add_subplot(1, n_figs, 1, projection='3d')
    _set_figure_labels(ax, '$\chi^2$')
    x1, x2, y = _get_target_points(x1_range=[0, 1], x2_range=[0, 1], func1=conf_weight, func2=identity)
    ax.plot_trisurf(x1, x2, y, linewidth=0.2, cmap=cm.jet)
    fig_pos += 1

    # plot the learned function in ranges 0-1
    ax = fig.add_subplot(1, n_figs, 2, projection='3d')
    _set_figure_labels(ax, '$GR$')
    x1, x2, y_ = _get_target_points(x1_range=[0, 1], x2_range=[0, 1], func1=gainratio, func2=identity)
    ax.plot_trisurf(x1, x2, y_, linewidth=0.2, cmap=cm.jet)
    ax.set_zlim(bottom=0, top=1)
    fig_pos += 1

    ax = fig.add_subplot(1, n_figs, 3, projection='3d')
    _set_figure_labels(ax, '$RF$')
    x1, x2, y_ = _get_target_points(x1_range=[0.001, 1], x2_range=[0.001, 1], ydiv=400, func1=rel_factor, func2=identity)
    ax.plot_trisurf(x1, x2, y_, linewidth=0.2, cmap=cm.jet, alpha=0.6)
    #ax.set_zlim(bottom=0, top=1)
    fig_pos += 1


    plotpath = os.path.join('../', 'chi_gr_rf.pdf')
    plt.tight_layout()
    plt.savefig(plotpath)


plot()
#data = DatasetLoader(dataset='ohsumed', vectorize='count', rep_mode='sparse', positive_cat=6)
#print data.num_features()

#from scipy.stats import t
#from scipy.stats import norm

