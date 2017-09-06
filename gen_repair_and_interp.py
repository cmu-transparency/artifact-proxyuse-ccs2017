import sys
sys.tracebacklimit=5

import numpy as np
from sklearn import tree
from sklearn import metrics
from pandas  import Series,DataFrame
from pymonad import *

from probmonad  import *
from util       import *
from ml_util    import *
from lang       import *
from conversion import *
from repair     import *

from plot_util import *
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
from scipy.interpolate import spline

e = experiment_from_args()
metric = e.association
proj   = nth(e.sensitive_index)

distData = lift(e.data.itertuples_noid())
distDataTest = lift(e.data_full.itertuples_noid())
distX    = lift(map(lambda s: State(s), e.dataX.itertuples_noid()))
distY    = lift(map(lambda s: State(s), e.dataY.itertuples_noid()))

error_base = 1.0 - distY.probmax()
error = rel_error(error_base)

exp = e.expression.simplify()

if e.verbose:
    eprint("whole model =\n%s\n" % tab(str(exp)))

eprint("max error     = %0.3f\n" % error_base)
error_cls = distData.expectation(sklearn_inaccurate(e.classifier,e.class_index))
eprint("error of tree = %0.3f(%0.3f%% of max)\n" % (error_cls,error(error_cls)))
error_exp = distData.expectation(exp_inaccurate(exp,e.class_index))
eprint("error of exp  = %0.3f(%0.3f%% of max)\n" % (error_exp,error(error_exp)))

utility = (lambda exp: distData.expectation(exp_accurate(exp,e.class_index)))

def evalfun(m):
    error_raw = distData.expectation(exp_inaccurate(m,e.class_index))
    error_rel = error(error_raw)
    size = m.size_()

    error_raw_test = distDataTest.expectation(exp_inaccurate(m,e.class_index))
    error_rel_test = error(error_raw)
    ret1 = "(train) utility=%0.8f(%0.8f%% of max error), model size=%d" % (1.0 - error_raw, error_rel,size)
    ret2 = "(test ) utility=%0.8f(%0.8f%% of max error)" % (1.0 - error_raw_test, error_rel_test)
    return "\n".join([ret1,ret2])

repair_function = repair
if e.subrepair:
    repair_function = sub_repair

epsilon = e.epsilon[0]
delta   = e.delta[0]

exp_r = repair_function(distX, nth(e.sensitive_index),
                        exp,
                        epsilon, delta,
                        e.association,
                        evalfun,
                        utility,
                        e.order
                        )
    
eprint("finally, " + evalfun(exp_r) + "\n")

error_repaired = distData.expectation(exp_inaccurate(exp_r,e.class_index))

##############

exp = exp_r.simplify().copy_() #e.expression.simplify().copy_()

epsilon_list  = []      # epsilon
delta_list    = []      # delta
size_list     = []      # size
arrows_list   = []
color_list    = []

# list of parent nodes
plist = []
threshold = e.epsilon[0]

model = e.args.model
ident = e.args.label

print >>e.handle1, "\t".join(['label','epsilon','delta','size','height','va','ha','xtext','ytext','color','model','ident'])
e.handle1.flush()
print >>e.handle3, "\t".join(['label','epsilon','delta','size','height','va','ha','xtext','ytext','color','model','ident'])
e.handle2.flush()
print >>e.handle2, "\t".join(['parent_epsilon','parent_delta','child_epsilon','child_delta','color','model','ident'])
e.handle3.flush()

print >>e.handle3, "\t".join(['B','0.0','0.0','10.0','10.0','baseline','right','18.0','18.0',e.color,model,ident])
e.handle3.flush()

considered = set()
count_unique = 0
count_duplicates = 0

def count(d,parent):
    global considered
    global count_unique
    global count_duplicates
    h = d.subholed.str_()
    if h in considered:
        #print "already seen this decomposition"
        count_duplicates += 1
        return
    else:
        count_unique += 1
        considered.add(h)
    
    for sd in d.get_immediate_subdecomps():
        if not (type(sd.submodel) is ExpConst):
            if not (type(sd.submodel) is ExpVar):
                count(sd, d)

def consider(d,parent,i,t):
    global considered
    h = d.subholed.str_()
    if h in considered:
        #print "already seen this decomposition"
        return
    else:
        considered.add(h)
    
    epsilon = d.get_epsilon(distX,metric)
    delta   = d.get_delta()

    epsilon_list.append(epsilon)
    delta_list  .append(delta)
    size_list   .append(d.submodel.size())
    plist       .append(i)
    color_list  .append(e.color)
    index = len(epsilon_list) - 1
    
    if parent is not None:
        pepsilon  = parent.get_epsilon(distX,metric)
        pdelta    = parent.get_delta()
        arrows_list.append(((pdelta,pepsilon),(delta,epsilon),delta - pdelta,d,parent))

        print >>e.handle2, "\t".join(map(str, [pepsilon,pdelta,epsilon,delta,e.color,model,ident]))
        e.handle2.flush()
    else:
        print >>e.handle3, "\t".join(['A','1.0',str(d.delta),'10.0','10.0','top','right','-18.0','-18.0',e.color,model,ident])
        e.handle3.flush()

    print t + type(d.submodel).__name__ 
    
    print >>e.handle1, "\t".join(map(str, [latexify(d.submodel.smallstr()).split("\n")[0],epsilon,delta,d.submodel.size(),d.submodel.height(),'baseline','right',-18.0,18.0,e.color,model,ident]))
    e.handle1.flush()
    
    for sd in d.get_immediate_subdecomps():
        # Skip constants since they are all the same point in the picture at (0,0).
        if not (type(sd.submodel) is ExpConst):
            if not (type(sd.submodel) is ExpVar):
                consider(sd, d, index, t + "  ")

exp.flow(distX, distX, 1.0, proj)
decomp = Decomposition(1, 1, proj, exp, ExpHole(0), exp, lens=lens_identity)

count(decomp, None)

print "unique = %d, duplicates = %d" % (count_unique, count_duplicates)

considered = set()

consider(decomp, None, 0, "")

e.close_handles()

# plot labels
plt.xlabel(r'$\delta$ / influence [probability]')
plt.ylabel(r'$\epsilon$ / association (%s)' % (metric))

plt.grid(b=True)
plt.autoscale(enable=True,tight=False,axis='both')

plt.scatter(delta_list, epsilon_list, s=size_list, color=color_list)

ax = plt.axes()
ax.set_yscale('log', basey=2)
ax.set_xscale('log', basex=2)
for (p1,p2,diff,d,dp) in arrows_list:
    color = e.color
        
    plt.quiver(p1[0],p1[1],p2[0]-p1[0],p2[1]-p1[1],units='dots',width=1,scale_units='xy',angles='xy',scale=1,color=color,alpha=0.25)

if e.save_figure is not None:
    plt.savefig(e.save_figure)

if e.show_figure:
    plt.show()
