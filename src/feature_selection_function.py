import math

def get_probs(tpr, fpr, pc):
    # tpr = p(t|c) = p(tp)/p(c) = p(tp)/(p(tp)+p(fn))
    # fpr = p(t|_c) = p(fp)/p(_c) = p(fp)/(p(fp)+p(tn))
    pnc = 1.0 - pc
    tp = tpr * pc
    fn = pc - tp
    fp = fpr * pnc
    tn = pnc - fp
    return ContTable(tp=tp, fn=fn, fp=fp, tn=tn)

def infogain(tpr, fpr, pc):
    cell = get_probs(tpr, fpr, pc)

    def ig_factor(p_tc, p_t, p_c):
        den = p_t * p_c
        if den != 0.0 and p_tc != 0:
            return p_tc * math.log(p_tc / den, 2)
        else:
            return 0.0

    return ig_factor(cell.p_tp(), cell.p_f(), cell.p_c()) + ig_factor(cell.p_fp(), cell.p_f(), cell.p_not_c()) \
           + ig_factor(cell.p_fn(), cell.p_not_f(), cell.p_c()) + ig_factor(cell.p_tn(), cell.p_not_f(), cell.p_not_c())

def gainratio(tpr, fpr, pc):
    pnc = 1.0 - pc
    norm = pc * math.log(pc, 2) + pnc * math.log(pnc, 2)
    return infogain(tpr, fpr, pc) / (-norm)

def chisquare(tpr, fpr, pc):
    cell = get_probs(tpr, fpr, pc)
    den = cell.p_f() * cell.p_not_f() * cell.p_c() * cell.p_not_c()
    if den==0.0: return 0.0
    num = gss(tpr,fpr,pc)**2
    return num / den

def rel_factor(tpr, fpr, pc):
    cell = get_probs(tpr, fpr, pc)

    a = cell.tp
    c = cell.fp
    if c == 0: c = 1

    return math.log(2.0 + (a * 1.0 / c), 2)

def idf(tpr, fpr, pc):
    cell = get_probs(tpr, fpr, pc)
    if cell.p_f()>0:
        return math.log(1.0 / cell.p_f())
    return 0.0


def gss(tpr, fpr, pc):
    cell = get_probs(tpr, fpr, pc)
    return cell.p_tp()*cell.p_tn() - cell.p_fp()*cell.p_fn()

class ContTable:
    def __init__(self, tp=0, tn=0, fp=0, fn=0):
        self.tp=tp
        self.tn=tn
        self.fp=fp
        self.fn=fn

    def get_d(self): return self.tp + self.tn + self.fp + self.fn

    def get_c(self): return self.tp + self.fn

    def get_not_c(self): return self.tn + self.fp

    def get_f(self): return self.tp + self.fp

    def get_not_f(self): return self.tn + self.fn

    def p_c(self): return (1.0*self.get_c())/self.get_d()

    def p_not_c(self): return 1.0-self.p_c()

    def p_f(self): return (1.0*self.get_f())/self.get_d()

    def p_not_f(self): return 1.0-self.p_f()

    def p_tp(self): return (1.0*self.tp) / self.get_d()

    def p_tn(self): return (1.0*self.tn) / self.get_d()

    def p_fp(self): return (1.0*self.fp) / self.get_d()

    def p_fn(self): return (1.0*self.fn) / self.get_d()

    def tpr(self):
        c = 1.0*self.get_c()
        return self.tp / c if c > 0.0 else 0.0

    def fpr(self):
        _c = 1.0*self.get_not_c()
        return self.fp / _c if _c > 0.0 else 0.0


def feature_label_contingency_table(positive_document_indexes, feature_document_indexes, nD):
    tp_ = len(positive_document_indexes & feature_document_indexes)
    fp_ = len(feature_document_indexes - positive_document_indexes)
    fn_ = len(positive_document_indexes - feature_document_indexes)
    tn_ = nD - (tp_ + fp_ + fn_)
    return ContTable(tp=tp_, tn=tn_, fp=fp_, fn=fn_)
