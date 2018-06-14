import numpy as np
from numpy import log, add, multiply, divide
from sklearn.linear_model import LogisticRegression

from data.dataset_loader import TextCollectionLoader
import cPickle as pickle
from utils.helpers import create_if_not_exists
import os
from scipy.sparse.linalg import norm
from scipy.sparse import issparse, csr_matrix
from joblib import Parallel
from joblib import delayed
import numpy as np
from sklearn.metrics import f1_score



ALLOW_DENSIFICATION = True

k1 = 1.2
k2 = 0.
k3 = 1000.
b = 0.75

# ----------------------------------------------------------------
# Athoms
# ----------------------------------------------------------------
#
# csr_matrix:
# ----------------------------------------------------------------
def ft01_csr(tf):
    return tf.copy()

def ft02_csr(tf):
    tf = tf.copy()
    nonzeros = (tf!=0)
    tf[nonzeros] = 1+log(tf[nonzeros])
    return tf

def ft03_csr(tf):
    tf = tf.copy()
    maxtf = tf.max(axis=1).toarray().flatten()
    rows,cols = tf.nonzero()
    tf[rows, cols] = 0.5 + ((0.5 + tf[rows, cols]) / maxtf[rows])
    return tf

def ft04_csr(tf):
    tf = tf.copy()
    avgtf = tf.sum(axis=1).getA1() / tf.getnnz(axis=1) # the .mean takes into account all 0 values
    lg_avgtf = np.log(avgtf)
    rows,cols = tf.nonzero()
    tf[rows,cols] = ((1. + log(tf[rows, cols])) / (1. + lg_avgtf[rows]))
    return tf

def ft05_csr(tf):
    tf = tf.copy()
    dl = tf.sum(axis=1).getA1()
    avgdl = dl.mean()
    rows,cols = tf.nonzero()
    tf[rows,cols] = ((k1+1)*tf[rows,cols]) / ( (k1*((1-b)+b*dl[rows]/avgdl) + dl[rows]) + tf[rows,cols])
    return tf

# dense row vector
# ---------------------------------
def __df(tf):
    #the clip is useful to prevent div.by zero, which could happen because the tf can be a split of the training set, so
    #it is not guaranteed to have all columns non-empty
    return (tf > 0).toarray().sum(axis=0).clip(1)

def ft06_row(tf):
    N = tf.shape[0]
    idf = log(float(N) / __df(tf))
    return idf.reshape(1,-1)
    #return np.tile(idf, (N,1))

def ft07_row(tf):#, notile=False):
    N = tf.shape[0]
    idf = log(1. + float(N) / __df(tf))
    return idf.reshape(1, -1)
    # if notile:
    #     return idf
    # else:
    #     return np.tile(idf, (N, 1))

def ft08_row(tf):
    N = tf.shape[0]
    idf = log((0.5 + float(N) - __df(tf))/0.5)
    # return np.tile(idf, (N, 1))
    return idf.reshape(1, -1)

def ft09_row(tf):
    N = tf.shape[0]
    df = __df(tf)
    idf = log((0.5 + float(N) - df)/(df + 0.5))
    # return np.tile(idf, (N, 1))
    return idf.reshape(1, -1)

def ft10_row(tf):
    N = tf.shape[0]
    df = __df(tf)
    idf = log((float(N) - df)/df)
    # return np.tile(idf, (N, 1))
    return idf.reshape(1, -1)

def ft11_row(tf):
    N = tf.shape[0]
    df = __df(tf)
    idf = log((float(N) + 0.5)/df) / log(N+1.)
    # return np.tile(idf, (N, 1))
    return idf.reshape(1, -1)

# dense column vector
# --------------------------------
def ft12_col(tf):
    tffactor = ft01_csr(tf)
    idffactor = ft07_row(tf).squeeze()
    rows,cols = tf.nonzero()
    tfidf = tffactor.copy()
    tfidf[rows,cols] = np.multiply(tffactor[rows,cols], idffactor[cols])
    # nF = tf.shape[1]
    cosnorm = 1. / norm(tfidf , ord=2, axis=1)
    # return np.tile(cosnorm.reshape(-1,1), (1, nF))
    return cosnorm.reshape(-1, 1)

def ft13_col(tf):
    tffactor = ft02_csr(tf)
    idffactor = ft07_row(tf).squeeze()
    rows,cols = tf.nonzero()
    tfidf = tffactor.copy()
    tfidf[rows,cols] = np.multiply(tffactor[rows,cols], idffactor[cols])
    # nF = tf.shape[1]
    cosnorm = 1. / norm(tfidf, ord=2, axis=1)
    # return np.tile(cosnorm.reshape(-1,1), (1, nF))
    return cosnorm.reshape(-1, 1)

def ft14_col(tf): # average document length in bytes
    dl = tf.sum(axis=1)*6. # the average number of chars per word is around 6
    return dl.reshape(-1, 1)


# helpers to obtain the best slope for the pivot-length normalization (t15, t16, and t17)
# --------------------------------
def _t15_denom(slope, t13, avgt13):
    return (1. - slope) + slope * (avgt13 / t13)

def _t16_denom(slope, dl, avgdl):
    return (1. - slope) * avgdl + slope * dl

def _t17_denom(slope, n_unique_terms, pivot):
    return (1. - slope) * pivot + slope * n_unique_terms

def _score_normalization_lr(X,y,Xva,yva):
    lr = LogisticRegression(C=1000) # some normalization functions work bad with C=1
    lr.fit(X, y)
    yva_ = lr.predict(Xva)
    return f1_score(yva, yva_)

def _slope_t15_eval(slope, Xtr, ytr, t13, avgt13, Xva, yva, t13va, avgt13va):
    Xtr_normalized = division(Xtr, _t15_denom(slope, t13, avgt13))
    Xva_normalized = division(Xva, _t15_denom(slope, t13va, avgt13va))
    return _score_normalization_lr(Xtr_normalized, ytr, Xva_normalized, yva)

def _slope_t16_eval(slope, Xtr, ytr, dl, avgdl, Xva, yva, dlva):
    Xtr_normalized = division(Xtr, _t16_denom(slope, dl, avgdl))
    Xva_normalized = division(Xva, _t16_denom(slope, dlva, avgdl))
    return _score_normalization_lr(Xtr_normalized, ytr, Xva_normalized, yva)

def _slope_t17_eval(slope, Xtr, ytr, n_unique_terms, Xva, yva, n_unique_terms_va):
    pivot_tr = n_unique_terms.mean()
    Xtr_normalized = division(Xtr, _t17_denom(slope, n_unique_terms, pivot_tr))
    pivot_va = n_unique_terms_va.mean()
    Xva_normalized = division(Xva, _t17_denom(slope, n_unique_terms_va, pivot_va))
    return _score_normalization_lr(Xtr_normalized, ytr, Xva_normalized, yva)

def find_best_slope_t15(Xtr, ytr, Xva, yva, slopes=np.arange(0.1, 1.1, 0.1)): #we skip 0 since this equates to doing nothing (so returning t01(tf))
    t13 = ft13_col(Xtr)
    avgt13 = t13.mean()

    t13va = ft13_col(Xva)
    avgt13va = t13va.mean()

    scores = Parallel(n_jobs=-1, backend="threading")(
        delayed(_slope_t15_eval)(slope, Xtr, ytr, t13, avgt13, Xva, yva, t13va, avgt13va) for slope in slopes
    )
    best_slope = slopes[np.argsort(scores)[-1]]

    return best_slope

def find_best_slope_t16(Xtr, ytr, Xva, yva, slopes=np.arange(0.1, 1.1, 0.1)): #we skip 0 since this equates to doing nothing (so returning t01(tf))
    dl = ft14_col(Xtr)
    avgdl = dl.mean()

    dlva = ft14_col(Xva)

    scores = Parallel(n_jobs=-1, backend="threading")(
        delayed(_slope_t16_eval)(slope, Xtr, ytr, dl, avgdl, Xva, yva, dlva) for slope in slopes
    )
    best_slope = slopes[np.argsort(scores)[-1]]

    return best_slope

def find_best_slope_t17(Xtr, ytr, Xva, yva, slopes=np.arange(0.1, 1.1, 0.1)): #we skip 0 since this equates to doing nothing (so returning t01(tf))
    n_unique_terms = (Xtr>0).sum(axis=1)
    n_unique_terms_va = (Xva > 0).sum(axis=1)

    scores = Parallel(n_jobs=-1, backend="threading")(
        delayed(_slope_t17_eval)(slope, Xtr, ytr, n_unique_terms, Xva, yva, n_unique_terms_va) for slope in slopes
    )
    best_slope = slopes[np.argsort(scores)[-1]]

    return best_slope

def ft15_col(tf, slope_t15):
    t13 = ft13_col(tf)
    avgt13 = t13.mean()
    return 1. / _t15_denom(slope_t15, t13, avgt13)

def ft16_col(tf, slope_t16):
    dl = ft14_col(tf)
    avgdl = dl.mean()
    return 1. / _t16_denom(slope_t16, dl, avgdl)

def ft17_col(tf, slope_t16):
    n_unique_terms = (tf > 0).sum(axis=1)
    return 1. / _t17_denom(slope_t16, n_unique_terms, pivot=n_unique_terms.mean())

def ft18_csr(tf):
    dl = ft14_col(tf)
    avgdl = dl.mean()
    factor = (k1*(1.-b)+b*(dl/avgdl))
    denom = addition(tf, factor)
    return division(1.,denom)

def ft19(tf): # not applicable to text classification
    pass
    # qtf = ? not applicable to text classification
    # numer = multiply(k3+1.,qtf)
    # denom = addition(qtf, k3)
    # return division(numer, denom)

def ft20(tf): # not applicable to text classification
    pass
    # qtf = ?
    # maxqtf = ?
    # rows,cols = qtf.nonzero()
    # qtf[rows, cols] = 0.5 + ((0.5 + qtf[rows, cols]) / maxqtf[rows])
    # return qtf

# constant random float in [0., 99.]
def ct21_float(tf=None):
    def _constant():
        return np.random.rand()*100.0
    return _constant


# ----------------------------------------------------------------
# Operations:
# ----------------------------------------------------------------

def logarithm(x):
    if issparse(x):
        if x.nnz > 0:
            x = x.copy()
            rows, cols = x.nonzero()
            x[rows, cols] = np.log(np.abs(x[rows,cols]))
            x.eliminate_zeros()
        return x

    else:
        return np.log(np.abs(x))

def addition(x, y):
    if isinstance(x, float): # float + ? ---
        if issparse(y):
            return addition(y,x)
        else: #y is float or any dense array
            return x+y
    elif issparse(x): # csr + ? ---
        if issparse(y): # csr + csr
            return x+y
        else:
            z = x.copy()
            rows, cols = x.nonzero()
            if isinstance(y, float):  # csr + float
                z[rows, cols] += y
            else: # csr + dense
                r,c = y.shape
                if r == 1: # csr + dense-row
                    z[rows, cols] += y[0, cols]
                elif c == 1: # csr + dense-col
                    z[rows, cols] += y[rows, 0].T
                else: # csr + dense-full
                    z[rows, cols] += y[rows, cols]
            return z
    else: # dense + ?
        if isinstance(y, float) or issparse(y):
            return addition(y, x)
        if x.shape == y.shape:
            return x+y
        else:
            if ALLOW_DENSIFICATION:
                xr, xc = x.shape
                yr, yc = y.shape
                nr, nc = max(xr,yr), max(xc, yc)
                x = __tile_to_shape(x, (nr, nc))
                y = __tile_to_shape(y, (nr, nc))
                return x+y
            else:
                raise ValueError('unallowed operation')


def __tile_to_shape(x, shape):
    nr, nc = shape
    r,c = x.shape
    if r==nr and c==nc:
        return x
    elif r==1 and c==nc:
        return np.tile(x, (nr, 1))
    elif r==nr and c==1:
        return np.tile(x, (1, nc))
    else:
        raise ValueError('format error')

def multiplication(x, y):
    if isinstance(x, float): # float * ? ---
        if issparse(y):
            return multiplication(y,x)
        else: #y is float or any dense array
            return np.multiply(x, y)
    elif issparse(x): # csr * ? ---
        if issparse(y): # csr * csr
            return x.multiply(y)
        else:
            z = x.copy()
            rows, cols = x.nonzero()
            if isinstance(y, float):  # csr * float
                z[rows, cols] *= y
            else: # csr * dense
                r,c = y.shape
                if r == 1: # csr * dense-row
                    z[rows, cols] = np.multiply(z[rows, cols], y[0, cols])
                elif c == 1: # csr * dense-col
                    z[rows, cols] = np.multiply(z[rows, cols], y[rows, 0].T)
                else: # csr * dense-full
                    z[rows, cols] = np.multiply(z[rows, cols], y[rows, cols])
            return z
    else: # dense * ?
        if isinstance(y, float) or issparse(y):
            return multiplication(y, x)
        if x.shape == y.shape:
            return np.multiply(x,y)
        else:
            if ALLOW_DENSIFICATION:
                xr, xc = x.shape
                yr, yc = y.shape
                nr, nc = max(xr,yr), max(xc, yc)
                x = __tile_to_shape(x, (nr, nc))
                y = __tile_to_shape(y, (nr, nc))
                return np.multiply(x,y)
            else:
                raise ValueError('unallowed operation')

def division(x, y):
    if isinstance(y,float) and y==0: raise ValueError('division by 0')
    if isinstance(x, float): # float / ? ---
        if x == 0.: return 0.
        if isinstance(y, float):
            return x/y
        if issparse(y):
            z = y.copy()
            rows, cols = z.nonzero()
            z[rows,cols] = x / z[rows,cols]
            return z
        else: #y is any dense array
            z = np.divide(x, y, where=y!=0)
            z[y==0] = 0. # where y==0 np places a 1
            return z
    elif issparse(x): # csr / ? ---
        if issparse(y): # csr / csr
            z = y.copy()
            rows, cols = y.nonzero()
            z[rows, cols] = np.divide(x[rows,cols],y[rows,cols]) #y[rows,cols] come from nonzero()
        else:
            z = x.copy()
            rows, cols = x.nonzero()
            if isinstance(y, float):  # csr / float
                z[rows, cols] = np.divide(x[rows, cols], y) # y is nonzero
            else: # csr / dense
                r,c = y.shape
                if r == 1: # csr / dense-row
                    denom = y[0, cols]
                elif c == 1: # csr / dense-col
                    denom = y[rows, 0].T
                else: # csr / dense-full
                    denom = y[rows, cols]
                denom = np.asarray(denom).flatten()
                zs = np.divide(x[rows, cols].flatten(), denom, where=denom!=0).reshape(1,-1)
                zs[0,denom==0] = 0. # where y==0 np places a 1
                z[rows, cols] = zs
        z.eliminate_zeros()
        return z
    else: # dense / ?
        if isinstance(y, float):
            return np.divide(x,y)
        elif issparse(y):
            z = y.copy()
            rows, cols = y.nonzero()
            r, c = x.shape
            if r == 1:  # dense-row / csr
                numer = x[0, cols]
            elif c == 1:  # dense-col / csr
                numer = x[rows, 0]
            else:  # dense-full / csr
                numer = x[rows, cols]
            denom = y[rows, cols]
            z[rows, cols] = np.divide(numer.flatten(), denom.flatten()) # denom comes from nonzero()
            z.eliminate_zeros()
            return z
        if x.shape == y.shape:
            zs = np.divide(x, y, where=y!=0)
            zs[y==0]=0.
            return zs
        else:
            if ALLOW_DENSIFICATION:
                xr, xc = x.shape
                yr, yc = y.shape
                nr, nc = max(xr,yr), max(xc, yc)
                x = __tile_to_shape(x, (nr, nc))
                y = __tile_to_shape(y, (nr, nc))
                return division(x,y)
            else:
                raise ValueError('unallowed operation')

# ----------------------------------------------------------------
# Collection Loader
# ----------------------------------------------------------------
def loadCollection(dataset, pos_cat, fs, data_home='../genetic_home'):
    version = TextCollectionLoader.version
    create_if_not_exists(data_home)
    pickle_name = '-'.join(map(str,[dataset,pos_cat,fs,version]))+'.pkl'
    pickle_path = os.path.join(data_home, pickle_name)
    if not os.path.exists(pickle_path):
        data = TextCollectionLoader(dataset=dataset, vectorizer='count', rep_mode='sparse', positive_cat=pos_cat,feat_sel=fs)
        Xtr, ytr = data.get_train_set()
        Xva, yva = data.get_validation_set()
        Xte, yte = data.get_test_set()
        pickle.dump((Xtr,ytr,Xva,yva,Xte,yte), open(pickle_path,'wb'), pickle.HIGHEST_PROTOCOL)
    else:
        Xtr, ytr, Xva, yva, Xte, yte = pickle.load(open(pickle_path,'rb'))
    Xtr=Xtr.asfptype()
    Xva=Xva.asfptype()
    Xte=Xte.asfptype()
    return Xtr,ytr,Xva,yva,Xte,yte


# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------
unary_function_list = [logarithm]
binary_function_list = [addition, multiplication, division]
param_free_terminals = [ft01_csr, ft02_csr, ft03_csr, ft04_csr, ft05_csr, ft06_row, ft07_row, ft08_row, ft09_row, ft10_row,
                 ft11_row, ft12_col, ft13_col, ft14_col, ft18_csr, ct21_float]
parametriced_terminals = [ft15_col, ft16_col, ft17_col]
terminal_list = param_free_terminals + parametriced_terminals

constant_list = map(float, range(100))

class Operation:
    def __init__(self, operation, nargs):
        assert hasattr(operation, '__name__'), 'anonymous operation '
        assert hasattr(operation, '__call__'), '{} is not callable'.format(operation.__name__)
        self.operation = operation
        self.nargs = nargs

    def __call__(self, *args, **kwargs):
        assert len(args)==self.nargs, 'wrong number of arguments'
        # if None in args:
        #
        #     return None
        return self.operation(*args)

    def __str__(self):
        return self.operation.__name__

class Terminal:
    def __init__(self, name, terminal):
        self.name = name
        self.terminal = terminal

    def __str__(self):
        if self.isconstant():
            return str(self.terminal)
        else:
            return self.name

    def isconstant(self):
        return self.name == ct21_float.__name__

class Tree:
    def __init__(self, node=None):
        assert isinstance(node, Operation) or isinstance(node, Terminal), 'invalid node'
        self.node = node
        self.fitness_score = None
        if isinstance(self.node, Operation):
            self.branches = [None] * self.node.nargs
        else:
            self.branches = []

    def add(self, *trees):
        assert [isinstance(t, Tree) for t in trees], 'unexpected type'
        assert len(trees) == len(self.branches), 'wrong number of branches'
        for i,tree in enumerate(trees):
            self.branches[i]=tree

    def depth(self):
        if isinstance(self.node, Terminal):
            return 1
        else:
            return 1+max(branch.depth() for branch in self.branches)

    def __str__(self):
        return self.__tostr(0) + '[depth={}]'.format(self.depth())

    def __tostr(self, tabs=0):
        _tabs = '\t'*tabs
        if isinstance(self.node, Terminal):
            return _tabs+str(self.node)
        return _tabs+'(' + str(self.node) +'\n'+ '\n'.join([b.__tostr(tabs+1) for b in self.branches]) + '\n'+_tabs+')'


    def __call__(self, eval_dict=None):
        """
        :param eval_dict: a dictionary of terminal-name:terminal, to be used (if passed) when evaluating the tree. This is
         useful to, e.g., evaluate the tree on the validation or test set
        :return:
        """
        if isinstance(self.node, Terminal):
            if eval_dict is None or self.node.isconstant(): #constants are to be taken from the tree
                return self.node.terminal
            else:
                return eval_dict[self.node.name].terminal
        else:
            args = []

            for t in self.branches:
                result = t(eval_dict)
                if result is None:
                    print('Node {} returned None'.format(t))
                args.append(result)
            ret = self.node(*args)
            # print('running ' + str(self))
            # if isinstance(ret, float):
            #     print('return {}'.format(ret))
            # else:
            #     print('return {} {}'.format(ret.shape, "sparse" if issparse(ret) else "dense"))
            return ret
            #return self.node(*[t(eval_dict) for t in self.branches])

    def fitness(self, eval_dict, ytr, yva):
        if self.fitness_score is None:
            #print('computing:')
            #print(self)
            Xtr_w = self()
            if isinstance(Xtr_w, float) or min(Xtr_w.shape) == 1: #non valid element, either a float, a row-vector or a colum-vector, not a full matrix
                self.fitness_score = 0
            else:
                Xva_w = self(eval_dict)
                #print('Training logistic regression...')
                logreg = LogisticRegression()
                logreg.fit(Xtr_w, ytr)
                #print('\tdone')
                yva_ = logreg.predict(Xva_w)

                self.fitness_score = f1_score(y_true=yva, y_pred=yva_, average='binary', pos_label=1)

        return self.fitness_score

    def preorder(self):
        import itertools
        return [self] + list(itertools.chain.from_iterable([branch.preorder() for branch in self.branches]))

    def random_branch(self):
        branches = self.preorder()
        if len(branches)>1:
            branches = branches[1:]
        return np.random.choice(branches)

    def copy(self):
        tree = Tree(self.node)
        if self.branches:
            tree.add(*[branch.copy() for branch in self.branches])
        tree.fitness_score = self.fitness_score
        return tree


def get_operations():
    return [Operation(f, 1) for f in unary_function_list] + [Operation(f, 2) for f in binary_function_list]


def get_terminals(X, slope_t15=None, slope_t16=None, slope_t17=None, asdict=False):
    terminals = [Terminal(f.__name__, f(X)) for f in param_free_terminals]
    if slope_t15: terminals.append(Terminal(ft15_col.__name__, ft15_col(X, slope_t15)))
    if slope_t16: terminals.append(Terminal(ft16_col.__name__, ft16_col(X, slope_t16)))
    if slope_t17: terminals.append(Terminal(ft17_col.__name__, ft17_col(X, slope_t17)))

    if asdict:
        terminals = {t.name:t for t in terminals}

    return terminals



def random_tree(max_length, exact_length, operation_pool, terminal_pool):
    """
    :param max_length: the maximun length of a branch
    :param exact_length: if True, forces all branches to have exactly max_length, if otherwise, a branch is only constrained
    to have <= max_length
    :return: a population
    """

    def __random_tree(length, exact_length, first_term=None):
        if first_term is None:
            term = np.random.choice(terminal_pool)
            if term.isconstant():  # the terminal is the constant function
                term = Terminal(term.name, term.terminal())  # which has to be instantiated to obtain the constant value
        else:
            term = first_term
        t = Tree(term)
        while t.depth() < length:
            op = np.random.choice(operation_pool)
            father = Tree(op)
            if op.nargs == 1:
                father.add(t)
            elif op.nargs == 2:
                branch_length = t.depth() if exact_length else np.random.randint(length - t.depth())
                branch = __random_tree(branch_length, exact_length)
                if np.random.rand() < 0.5:
                    father.add(t, branch)
                else:
                    father.add(branch, t)
            t = father
        return t

    first_term = np.random.choice(terminal_pool[:5]) # guarantee a full matrix is in the tree
    length = max_length if exact_length else np.random.randint(1, max_length+1)
    return __random_tree(length, exact_length, first_term)


def ramped_half_and_half_method(n, max_depth, operation_pool, terminal_pool):
    half_exact = [random_tree(max_depth, True, operation_pool, terminal_pool) for _ in range(n//2)]
    half_randlength = [random_tree(max_depth, False, operation_pool, terminal_pool) for _ in range(n//2)]
    return half_exact+half_randlength

def fitness_wrap(individual, ter_validation, ytr, yva):
    individual.fitness(ter_validation, ytr, yva)

def fitness_population(population, ter_validation, ytr, yva, n_jobs=-1, show=False):
    Parallel(n_jobs=n_jobs, backend="threading")(delayed(fitness_wrap)(ind,ter_validation, ytr, yva) for ind in population)
    sort_by_fitness(population)
    best = population[0]
    if show:
        for i, p in enumerate(population):
             print("{}: fitness={:.3f} [depth={}]".format(i, p.fitness_score, p.depth()))
        best_score = population[0].fitness_score
        print('Best individual score: {:.3f}'.format(best_score))
        print(best)
    return best

def sort_by_fitness(sorted_population):
    sorted_population.sort(key=lambda x: x.fitness_score, reverse=True)

def reproduction(population, rate_r=0.05):
    sort_by_fitness(population)
    totake = int(len(population) * rate_r)
    return population[:totake]

def mutate(population, operation_pool, terminal_pool, rate_m=0.05):
    length = int(len(population) * rate_m)
    mutated = [mutation(x, operation_pool, terminal_pool) for x in np.random.choice(population, length, replace=True)]
    return mutated

def mutation(x, operation_pool, terminal_pool):
    mutated = x.copy()
    old_branch = mutated.random_branch()
    mut_branch = random_tree(old_branch.depth(), False, operation_pool, terminal_pool)

    old_branch.node = mut_branch.node
    old_branch.branches = mut_branch.branches
    mutated.fitness_score = None

    return mutated

def cross(x,y):
    x_child = x.copy()
    y_child = y.copy()

    b1 = x_child.random_branch()
    b2 = y_child.random_branch()

    #swap attributes
    b1.node, b2.node = b2.node, b1.node
    b1.branches, b2.branches = b2.branches, b1.branches
    x_child.fitness_score, y_child.fitness_core = None, None

    return x_child, y_child


def crossover(population, rate_c=0.9, k=6): #tournament selection
    def tournament():
        group = np.random.choice(population, size=k, replace=True).tolist()
        group.sort(key=lambda x: x.fitness_score, reverse=True)
        return group[0]

    length = int(len(population)*rate_c)
    new_population = []
    while len(new_population) < length:
        parent1 = tournament()
        parent2 = tournament()
        child1, child2 = cross(parent1, parent2)
        new_population.append(child1)
        new_population.append(child2)

    return new_population



if __name__ == '__main__':
    dataset = 'ohsumed'
    pos_cat = 0
    fs = 0.1
    Xtr, ytr, Xva, yva, Xte, yte = loadCollection(dataset, pos_cat, fs)

    sparse_mat = ft01(Xtr)
    col = ft06(Xtr)
    row = ft13(Xtr)
    num = 20.

    tested = 0
    failed = 0

    def test_op(x,y,op=division):
        global tested, failed
        try:
            z = op(x,y)
            tested += 1
        except ValueError:
            z = None
            failed+=1
        print('tested',tested,'failed',failed)
        return z

    #dense = addition(row,col)

    c = test_op(sparse_mat,sparse_mat)
    c = test_op(sparse_mat,col)
    c = test_op(sparse_mat,row)
    c = test_op(sparse_mat,num)
    #c = test_op(sparse_mat,dense)

    c = test_op(row,num)
    c = test_op(row,sparse_mat)
    #c = test_op(row,dense)
    c = test_op(row,row)

    c = test_op(col,num)
    c = test_op(col,sparse_mat)
    #c = test_op(col,dense)
    c = test_op(col,row)
    c = test_op(col,col)

    # c = test_op(dense,num)
    # c = test_op(dense,sparse_mat)
    # c = test_op(dense,dense)
    # c = test_op(dense,row)
    # c = test_op(dense,col)





