import numpy as np
from scipy.stats import wilcoxon, ttest_rel

result_file ='/home/moreo/learning_to_weight/results/statistical_significance/Macrof1.txt'
#result_file ='/home/moreo/learning_to_weight/results/statistical_significance/microf1.txt'
ttest = wilcoxon
#ttest = ttest_rel

results = open(result_file, 'r').readlines()
results = [r.strip().split('\t') for r in results]
methods = [r[0] for r in results] # to preserve the order
results = {r[0]:np.array([float(x) for x in r[1:]]) for r in results}
print(results)

averages = {method:np.mean(scores) for method,scores in results.items()}
print(averages)
print(methods)

dag = '$\\dag$'
ddag = '$\\dag\\dag$'
hline = ' \\\\\\hline\n'
hhline = ' \\\\\\hline\\hline\n'
with open(result_file.replace('.txt','.tex'), 'w') as fo:
    n_methods = len(methods)
    fo.write('\\begin{tabular}{|c||'+'c|'*n_methods+'}' + hline)
    fo.write(' & '.join([' '] + ['\\begin{sideways}'+m+'\\end{sideways}' for m in methods]) + hhline)
    for i,mi in enumerate(methods):
        str_builder = [mi]
        for j,mj in enumerate(methods):
            if i==j:
                str_builder.append(' - ')
            else:
                _,pvalue = ttest(results[mi], results[mj])
                print(mi,mj,pvalue)
                sym = ' '
                if averages[mi] > averages[mj]:
                    #str_builder.append('%.5f' % pvalue)
                    if pvalue < 0.005:
                        sym = ddag
                    elif pvalue < 0.05:
                        sym = dag
                str_builder.append(sym)

        fo.write(' & '.join(str_builder) + hline)
    fo.write('\\end{tabular}')



