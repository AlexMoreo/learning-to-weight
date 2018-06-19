from scipy.sparse import issparse
import numpy as np


ALLOW_DENSIFICATION = True

# ----------------------------------------------------------------
# Operations:
# ----------------------------------------------------------------

def logarithm(x):
    if issparse(x):
        if x.nnz > 0:
            if x.nnz == 0: return x
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
            if x.nnz == 0: return x
            z = x.copy()
            rows, cols = x.nonzero()
            if isinstance(y, float):  # csr + float
                z[rows, cols] += y
            else: # csr + dense
                r,c = y.shape
                if r == 1: # csr + dense-row
                    y_ = y[0, cols]
                elif c == 1: # csr + dense-col
                    y_ = y[rows, 0]
                else: # csr + dense-full
                    y_ = y[rows, cols]
                z[rows, cols] += np.asarray(y_).flatten()
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
            if x.nnz == 0: return x
            z = x.copy()
            rows, cols = x.nonzero()
            if isinstance(y, float):  # csr * float
                z[rows, cols] *= y
            else: # csr * dense
                r,c = y.shape
                if r == 1: # csr * dense-row
                    y_ = y[0, cols]
                elif c == 1: # csr * dense-col
                    y_ = y[rows, 0]
                else: # csr * dense-full
                    y_ = y[rows, cols]
                z[rows, cols] = np.multiply(z[rows, cols], np.asarray(y_).flatten())
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
            if y.nnz == 0: return x
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
            if y.nnz == 0: return x
            z = y.copy()
            rows, cols = y.nonzero()
            z[rows, cols] = np.divide(x[rows,cols],y[rows,cols]) #y[rows,cols] come from nonzero()
        else:
            if x.nnz == 0: return x
            z = x.copy()
            rows, cols = x.nonzero()
            if isinstance(y, float):  # csr / float
                z[rows, cols] = np.divide(x[rows, cols], y) # y is nonzero
            else: # csr / dense
                r,c = y.shape
                if r == 1: # csr / dense-row
                    denom = y[0, cols]
                elif c == 1: # csr / dense-col
                    denom = y[rows, 0]
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
            if y.nnz == 0: return x
            z = y.copy()
            rows, cols = y.nonzero()
            r, c = x.shape
            if r == 1:  # dense-row / csr
                numer = x[0, cols]
            elif c == 1:  # dense-col / csr
                numer = x[rows, 0]
            else:  # dense-full / csr
                numer = x[rows, cols]
            numer = np.asarray(numer).flatten()
            denom = np.asarray(y[rows, cols]).flatten()
            z[rows, cols] = np.divide(numer, denom) # denom comes from nonzero()
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
