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

params = [(0.01,0.01)]

plots_x = {}
plots_y = {}

for param in params:
        plots_x[param] = []
        plots_y[param] = []

rows = [r[1] for d in e.data1 for r in d.iterrows()]
rows = filter(lambda r: (r['epsilon'],r['delta']) in params, rows)
rows.sort(key = lambda r: r['mixed_delta'])

for r in rows:
        param = (r['epsilon'],r['delta'])
        
        plots_x[param].append(r['mixed_delta'])
        plots_y[param].append(r['repaired_util'])

plt.xlabel(r'influence [probability]')
plt.ylabel(r'accuracy [ratio]')

plt.grid(True)

lines = []

for param in params:
        print "plotting %s" % str(param)
        plot_x = plots_x[param]
        plot_y = plots_y[param]

        plt.plot(plot_x,plot_y,'bo', color='black', markersize=4)

plt.axes().set_xlim(-0.02,0.45)
plt.axes().set_ylim(0.50,1.05)

plt.tight_layout()

if c.output is not None:
        print "saving figure to " + c.output
        plt.savefig(c.output)

if c.show:
        plt.show()
