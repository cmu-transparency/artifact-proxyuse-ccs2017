# coding=utf-8

# Subterm statistics. Given a dataset, sensitive attribute, class
# attribute, and classifier parameters, trains the classifier to
# predict class attribute. Then for each sub-expression in the
# resulting classifier, provides normalized mutual information and
# influence metrics.

import sys
reload(sys)
sys.setdefaultencoding('utf8')
sys.tracebacklimit=2

from detect  import *
from util    import *
from ml_util import *

from plot_util import *
import matplotlib.pyplot as plt

e = generated_from_args()
c = e.args

plots_x = []
plots_y = []

for d in e.data1:
        plot_x = []
        plot_y = []

        for r_temp in d.iterrows():
                r = r_temp[1]
                plot_x.append(r['dataset_size'])
                plot_y.append(r['runtime'])

        plots_x.append(plot_x)
        plots_y.append(plot_y)

plt.xlabel(r'dataset size [count]')
plt.ylabel(r'real runtime [s]')
plt.grid(True)

labels = ['tree','forest','logistic']
handles = []
markers = ['-bo', '-b+', '-bx']

for (plot_x,plot_y) in zip(plots_x, plots_y):
        i = plots_y.index(plot_y)
        line1, = plt.semilogy(plot_x,plot_y,'bo-',color=str(float(i)/len(plots_y)),label=labels[i],linewidth=2-0.5*i)
        #plt.semilogy(plot_x,plot_y,'bo',color=str(float(i)/len(plots_y)))
        handles.append(line1)

#plt.autoscale(enable=True,tight=False,axis='both')

#plt.legend(handles=handles,loc='upper left',handlelength=1, frameon=True)

plt.axes().set_xlim(0,max(lists_flatten(plots_x))*1.1)
plt.axes().set_ylim(0,max(lists_flatten(plots_y))*10)

plt.tight_layout()

if c.output is not None:
        print "saving figure to " + c.output
        plt.savefig(c.output)

if c.show:
        plt.show()
