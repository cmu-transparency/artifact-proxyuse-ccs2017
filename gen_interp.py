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

from detect  import all_stats,Decomposition
from util    import *
from ml_util import *
import sklearn.cross_validation as cross_validation

from plot_util import *
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
from scipy.interpolate import spline

e = experiment_from_args()
metric = e.association
proj   = nth(e.sensitive_index)

distData      = lift(e.data.itertuples_noid())
eprint("len(data)=%d, len(data_test)=%d\n" % (len(list(e.data.itertuples_noid())), len(list(e.data_test.itertuples_noid()))))
distData_test = lift(e.data_test.itertuples_noid())
distX         = lift(map(lambda s: State(s), e.dataX.itertuples_noid()))

utility      = (lambda exp: distData     .expectation(exp_accurate(exp,e.class_index)))
utility_test = (lambda exp: distData_test.expectation(exp_accurate(exp,e.class_index)))
max_utility      = utility(e.expression)
max_utility_test = utility_test(e.expression)
print "(train) utility = %f" % max_utility
print "(test ) utility = %f" % max_utility_test

# Lists used in storing epsilon and delta values
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

print >>e.handle3, "\t".join(['B','0.0','0.0','10.0','10.0','baseline','right','18.0','18.0',e.color,'nomodel','noident'])
e.handle3.flush()

counted    = set()
considered = dict()
count_unique = 0
count_duplicates = 0

def count(d,parent):
    global considered
    global counted
    global count_unique
    global count_duplicates
    h = d.subholed.str_()
    if h in counted:
        count_duplicates += 1
        return
    else:
        count_unique += 1
        counted.add(h)
    
    for sd in d.get_immediate_subdecomps():
        if not (type(sd.submodel) is ExpConst):
            if not (type(sd.submodel) is ExpVar):
                count(sd, d)

def consider(d,parent,i,t):
    global considered

    eprint(t)

    
    h = d.subholed.str_()
    if h in considered:
        eprint("(  seen) ")

        eprint(type(d.submodel).__name__ + " ... ")
        sys.stderr.flush()
        
        index = len(epsilon_list) - 1        
        (epsilon, delta) = considered[h]

    else:
        eprint("(unseen) ")

        eprint(type(d.submodel).__name__ + " ... ")
        sys.stderr.flush()

        
        epsilon = d.get_epsilon(distX,metric)
        delta   = d.get_delta()

        epsilon_list.append(epsilon)
        delta_list  .append(delta)
        size_list   .append(d.submodel.size())
        plist       .append(i)
        color_list  .append(e.color)
        index = len(epsilon_list) - 1

        considered[h] = (epsilon,delta)

        
    if parent is not None:
        pepsilon  = parent.get_epsilon(distX,metric)
        pdelta    = parent.get_delta()
        arrows_list.append(((pdelta,pepsilon),(delta,epsilon),delta - pdelta,d,parent))

        print >>e.handle2, "\t".join(map(str, [pepsilon,pdelta,epsilon,delta,e.color,model,ident]))
        e.handle2.flush()
    else:
        print >>e.handle3, "\t".join(['A','1.0',str(d.delta),'10.0','10.0','top','right','-18.0','-18.0',e.color,'nomodel','noident'])
        e.handle3.flush()

    print >>e.handle1, "\t".join(map(str, [latexify(d.submodel.smallstr()).split("\n")[0],epsilon,delta,d.submodel.size(),d.submodel.height(),'baseline','right',-18.0,18.0,e.color,model,ident]))
    e.handle1.flush()

    eprint("\n")
    
    for sd in d.get_immediate_subdecomps():
        # Skip constants since they are all the same point in the picture at (0,0).
        if not (type(sd.submodel) is ExpConst):
            if not (type(sd.submodel) is ExpVar):
                consider(sd, d, index, t + "  ")

#print e.expression

e.expression.flow(distX, distX, 1.0, proj)
decomp = Decomposition(1, 1, proj, e.expression, ExpHole(0), e.expression, lens=lens_identity)

count(decomp, None)

print "unique = %d, duplicates = %d" % (count_unique, count_duplicates)

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
