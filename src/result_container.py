import sys, os
from helpers import *

class ResultSet:
    def __init__(self, result_dir):
        self._container_name = '.result_container.dat'
        self._container_dir = result_dir
        create_if_not_exists(result_dir)
        #self._result_path = os.path.join(self._container_dir, self._container_path)
        #mode = 'a+' if os.path.exists(self._result_path) else 'w'
        self._container_file = open(os.path.join(self._container_dir, self._container_name), 'a+')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._container_file.close()

    def add(self, row, column, value, metric=None):
        #if (not row in self._container) or (not column in self._container[row]) or (self._container[row][column] != value):
        register = '%s;%s;%f;%s\n' % (row, column, value, metric)
        self._container_file.write(register)
        self._container_file.flush()
        #if row not in self._container:
        #    self._container[row] = dict()
        #self._container[row][column] = value

    def generate_report(self, title, rows=None, cols=None, metrics=None, path=None, comments=''):
        registers = dict()
        valid_rows, valid_cols, valid_metrics = set(), set(), set()

        with open(os.path.join(self._container_dir, self._container_name), 'r') as container:
            for register in container:
                row, col, val, metric = register[:-1].split(';')

                if rows and row not in rows: continue
                else: valid_rows.add(row)

                if cols and col not in cols: continue
                else: valid_cols.add(col)

                if metrics and metric not in metrics: continue
                else: valid_metrics.add(metric)

                registers[(row, col, metric)] = float(val)

        valid_rows = sorted(valid_rows)
        valid_cols = sorted(valid_cols)
        valid_metrics = sorted(valid_metrics)

        if not path:
            path = self._container_dir
        with open(os.path.join(path, title), 'w') as report:
            # write comments, if any
            if comments:
                report.write(comments.strip() + '\n')

            for metric in valid_metrics:
                #write column names
                if metric != 'None':
                    report.write('['+metric+']')
                for col in valid_cols:
                    report.write('\t'+str(col))
                report.write('\n')
                for row in valid_rows:
                    report.write(str(row))
                    for col in valid_cols:
                        index = (row, col, metric)
                        if index in registers:
                            val = registers[index]
                            report.write('\t%.3f' % val)
                        else:
                            report.write('\t-')
                    report.write('\n')
                report.write('\n')


with ResultSet(result_dir='res_test') as results:
    results.add('MetodoF', 'Cat0', 9.00, metric='Acc')
    results.add('MetodoG', 'Cat1', 0.01, metric='Acc')
    results.add('MetodoH', 'Cat3', 0.03, metric='F1')
    results.add('MetodoI', 'Cat1', 0.11, metric='F1')
    results.add('MetodoB', 'Cat3', 0.13, metric='Acc')
    results.add('MetodoB', 'Cat3', 10.13, metric='F1')
    results.add('MetodoC', 'Cat4', 0.24, metric='F1')
    results.add('MetodoD', 'Cat4', 0.34, metric='Acc')
    results.generate_report('FullReport')


