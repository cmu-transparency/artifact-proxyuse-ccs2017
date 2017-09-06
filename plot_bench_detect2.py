# coding=utf-8

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

plots_x  = []
plots_y1 = []
plots_y2 = []

for d in e.data1:
        plot_x  = []
        plot_y1 = []
        plot_y2 = []
        for r_temp in d.iterrows():
                r = r_temp[1]
                plot_x .append(r['sub_expressions'])
                plot_y1.append(r['runtime1'])
                plot_y2.append(r['runtime2'])
        plots_x .append(plot_x)
        plots_y1.append(plot_y1)
        plots_y2.append(plot_y2)

plt.xlabel(r'decompositions [count]')
plt.ylabel(r'real runtime [s]')
plt.grid(True)

handles = []

labels = ['tree','forest','logistic']
markers = ['-bo', '-b+', '-bx']

for (plot_x,plot_y1,plot_y2) in zip(plots_x, plots_y1, plots_y2):
        i = plots_x.index(plot_x)
        line1, = plt.loglog(plot_x,plot_y1,markers[i],color='black', label=labels[i]+' w infl.', linewidth=2)
        line2, = plt.loglog(plot_x,plot_y2,markers[i],color='gray', label=labels[i]+' w/o infl.', linewidth=1)

        handles.append(line1)
        handles.append(line2)

plt.legend(handles=handles,loc='upper left',handlelength=2, frameon=True)

plt.tight_layout()

if c.output is not None:
        print "saving figure to " + c.output
        plt.savefig(c.output)

if c.show:
        plt.show()
