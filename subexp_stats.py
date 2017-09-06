# coding=utf-8

# Subterm statistics. Given a dataset, sensitive attribute, class
# attribute, and classifier parameters, trains the classifier to
# predict class attribute. Then for each sub-expression in the
# resulting classifier, provides normalized mutual information and
# influence metrics.

# Comment out this line to not use cython
#import pyximport; pyximport.install()

import sys
reload(sys)
sys.setdefaultencoding('utf8')
sys.tracebacklimit=2

from detect  import all_stats
from util    import *
from ml_util import *
import sklearn.cross_validation as cross_validation

from plot_util import *
import matplotlib.pyplot as plt
from scipy.interpolate import spline

def outitem(i):
        #if type(i) is float:
        #    return "%0.16f" % (i)
        #else:
            return str(i)

def res(p):
    return (math.floor(20.0*p[0])/20.0,
            math.floor(20.0*p[1])/20.0)

# helper function which annotates plot above epsilon and delta threshold
def annotate_submodels(stats):

    texts = dict()

    epsilon = stats['epsilon']
    delta = stats['delta']

    smallstr = stats['submodel'].smallstr()

    metrics_vals = map(lambda m: stats['epsilon'][m], metrics)

    if e.verbose:
        #print "\t".join(map(outitem,metrics_vals+
        #                    [delta,stats['holes'],stats['max_holes'],stats['size'],stats['height'],stats['depth'],"\"" + smallstr + "\""
        #                    ]))
        eprint("\t".join(
                         map(outitem, metrics_vals +
                             [delta,stats['holes'],stats['max_holes'],stats['size'],stats['height'],"\"" + smallstr + "\""]
                            )
                        ) + "\n")

    #eprint(str(stats['submodel']))

    decomp = stats['decomp']

    if delta >= e.delta[0] and epsilon[metric] >= e.epsilon[0]:

        util = decomp.get_utility(utility)

        pos  = (delta,epsilon[metric])
        rpos = res(pos)
        offset = 0.0
        if rpos in texts:
            offset = texts[rpos]-15.0

        texts[rpos] = offset

        plt.annotate(latexify(smallstr) + ("\n(acc. - %0.3f)" % (max_utility - util)) ,
                    fontsize=9,
                    xycoords='data',
                    xy = pos,
                    textcoords='offset points',
                    xytext=(20.0,8.0+offset),
                    backgroundcolor='white',
                    va ='center',
                    ha ='left',
                    )

        plt.annotate("",
                    xycoords='data',
                    xy = pos,
                    textcoords='offset points',
                    xytext=(20.0,10.0+offset),
                    #backgroundcolor='blue',
                    va ='center',
                    ha ='left',
                    arrowprops=dict(arrowstyle='-',
                                    shrinkA=5.0,
                                    shrinkB=5.0,
                                    facecolor='gray',
                                    edgecolor='gray',
                                    ),
                    )

#def main():

e = experiment_from_args()

# passed arguments
metric = e.association
metrics = e.metrics
metrics_cols = map(lambda s: "ε-" + s, metrics)

prng = np.random.RandomState(e.seed)

if e.verbose:
    eprint("exp = " + str(e.expression) + "\n")
    eprint("done")

# number of iterations per split
iteration = e.iteration
if e.validation:
    ratio = [float(i+1)/10 for i in range(10)]
    instance = map(lambda s: len(e.dataX)*s, ratio)
    #ratio = [0.005,0.01,0.05,0.1,0.5,0.6,0.7,0.8,0.9,1]
else:
    ratio = [1]
j = 0

distData = lift(e.data.itertuples_noid())
distX    = lift(map(lambda s: State(s), e.dataX.itertuples_noid()))

# Lists used in storing epsilon and delta values
epsilon_list = []        # epsilon
delta_list = []          # delta
size_list = []           # size
submodel_list = []      # submodels
submodel_list_v = []      # submodels

# utility
utility = (lambda exp: distData.expectation(exp_accurate(exp,e.class_index)))
max_utility = utility(e.expression)

#left  = e.expression.lhs
#right = e.expression.rhs
##print left
##print e.cols
#
#def evalexp(x):
#    #print x
#    print left.eval(x), right.eval(x), e.expression.eval(x)
#    return singleton(True)
#distX >> evalexp
#exit(1)

# plot labels
plt.xlabel(r'$\delta$ / influence [probability]')
plt.ylabel(r'$\epsilon$ / association (%s)' % (metric))

# iterate over the ratio
while j < len(ratio):

    if e.validation:
        print "validation " + str(j+1) + " - split ratio " + str(ratio[j])

    i = 0

    data_epsilon = []
    data_delta = []
    data_size = []

    # iterate over the number of iteration
    while i < iteration:

        if e.validation:
           print "iteration " + str(i+1)

        # for each iteration sample randomly from the original dataset
        if ratio[j] == 1.0:
            dataX = e.dataX
        else:
            dataX, dataX_rest, dataY, dataY_rest = cross_validation.train_test_split(e.dataX, e.dataY, train_size=ratio[j], random_state = prng)

        # constructing distributions
        if e.verbose: eprint("constructing distributions ... ")
        distX = lift(map(lambda s: State(s), dataX.itertuples_noid()))
        if e.verbose: eprint("done\n")

        # start
        print "starting ... "

        all_submodels = all_stats(distX, nth(e.sensitive_index), e.expression, e.order)
        #all_submodels.sort(key = lambda s: -s[0][metric])

        if e.verbose:   print "\t".join(metrics_cols + ['δ','num_holes','max_num_holes','subexp_size','subexp_height','subexp_depth','subexp_start'])

        stat_epsilon = []
        stat_delta = []
        stat_size = []

        num_submodels = 0
        for stats in all_submodels:
            num_submodels = num_submodels+1
            stat_epsilon.append(stats['epsilon'][metric])
            stat_delta.append(stats['delta'])
            stat_size.append(stats['size'])

            if ratio[j] == 1.0:
                annotate_submodels(stats)
                submodel_list.append(stats['submodel'].smallstr())
                submodel_list_v.append(stats['submodel'].str_())
                i = iteration

        data_epsilon.append(stat_epsilon)
        data_delta.append(stat_delta)
        data_size.append(stat_size)

        i = i + 1

    print "found %d submodels" % num_submodels
    #rate_epsilon = map(lambda i: i/e.epsilon[0], data_epsilon)
    #rate_delta = map(lambda i: i/e.delta[0], data_delta)

    # append calculated epsilon and delta
    epsilon_list.append(data_epsilon)
    delta_list.append(data_delta)
    size_list.append(data_size)

    #for (delta, epsilon, size) in zip(data_delta,data_epsilon,data_size):
    plt.scatter(data_delta[0],data_epsilon[0],s=data_size[0],color='black',alpha=ratio[j])
    plt.axes().set_ylim([0,max(data_epsilon[0])*1.1])
    plt.axes().set_xlim([0,max(data_delta[0])*1.1])
    plt.grid(True)

    plt.autoscale(enable=True,tight=False,axis='both')
    #plt.axes().set_aspect('equal', 'datalim')
    #plt.tight_layout()

    j = j + 1

proxies = []
epsilons = []
deltas = []

i = 0
# detect a proxy usage
for (epsilon, delta) in zip(epsilon_list[-1][0], delta_list[-1][0]):
    if delta >= e.delta[0] and epsilon >= e.epsilon[0]:
        epsilons.append(epsilon)
        deltas.append(delta)
        proxies.append(i)
    i = i + 1

#print delta_list

print "######## PROXIES #########"
print "\n".join([submodel_list_v[i] for i in proxies])

# if there was no proxy found
if proxies == []:
    print "No proxy!"


# if proxies were found, plot them
elif e.validation:
    print proxies

    i = 0
    for target in proxies:

        fig = plt.figure(i+2)
        fig.suptitle('submodel %d - ' % target + latexify(submodel_list[target]))
        ax1 = plt.subplot(211)
        ax2 = plt.subplot(212)

        med_epsilon = []
        high_epsilon = []
        low_epsilon = []
        med_delta = []
        high_delta = []
        low_delta = []

        j = 0
        for (data_epsilon, data_delta) in zip(epsilon_list, delta_list):
            epsilon = sorted(map(lambda s: s[target], data_epsilon))
            delta = sorted(map(lambda s: s[target], data_delta))
            ax1.scatter([ratio[j]]*len(epsilon), epsilon, color='blue', s=2)
            ax2.scatter([ratio[j]]*len(delta), delta, color='red', s=2)

            med_epsilon.append(epsilon[int(len(epsilon)*0.5)])
            high_epsilon.append(epsilon[int(len(epsilon)*0.95)])
            low_epsilon.append(epsilon[int(len(epsilon)*0.05)])
            med_delta.append(delta[int(len(delta)*0.5)])
            high_delta.append(delta[int(len(delta)*0.95)])
            low_delta.append(delta[int(len(delta)*0.05)])

            j = j + 1

        ratio_smooth = np.linspace(min(ratio),max(ratio),200)
        med_epsilon = spline(ratio, med_epsilon, ratio_smooth)
        high_epsilon = spline(ratio, high_epsilon, ratio_smooth)
        low_epsilon = spline(ratio, low_epsilon, ratio_smooth)
        med_delta = spline(ratio, med_delta, ratio_smooth)
        high_delta = spline(ratio, high_delta, ratio_smooth)
        low_delta = spline(ratio, low_delta, ratio_smooth)


        # plot high and low values
        ax1.plot(ratio_smooth, high_epsilon, 'b--', ratio_smooth, low_epsilon, 'b--', alpha=0.5)
        ax2.plot(ratio_smooth, high_delta, 'r--', ratio_smooth, low_delta, 'r--', alpha=0.5)

        #ax1.plot(ratio_smooth, med_epsilon, 'b--', alpha=0.5)
        #ax2.plot(ratio_smooth, med_delta, 'r--', alpha=0.5)

        #ax1.set_yticks([epsilons[0]])
        #ax1.set_yticklabels([r'$\epsilon$'])

        # set labels
        ax1.set_xlabel('test / entire dataset ratio')
        ax1.set_ylabel(r'$\epsilon$ / association (%s)' % (metric))
        ax1.set_xlim([0,1])
        ax1.set_ylim(bottom=0)

        ax2.set_xlabel('test / entire dataset ratio')
        ax2.set_ylabel(r'$\delta$ / influence [probability]')
        ax2.set_xlim([0,1])
        ax2.set_ylim(bottom=0)

        # horizontal lines which indicates epsilons
        ax1.axhline(y=epsilons[i], color='blue', alpha=0.2, ls='dashed')
        ax1.axhline(y=e.epsilon[0], color='black', alpha=0.5, ls='dashed')

        ax2.axhline(y=deltas[i], color='red', alpha=0.2, ls='dashed')
        ax2.axhline(y=e.delta[0], color='black', alpha=0.5, ls='dashed')

        i = i + 1

        #plt.autoscale(enable=True,tight=False,axis='both')
        #plt.axes().set_aspect('equal', 'datalim')
        #plt.tight_layout()

## lists to store difference in epsilon and delta
#epsilon_diff = []
#delta_diff = []
#
#for (epsilons, deltas) in zip(epsilon_list, delta_list):
#    epsilon_diff.append(lift(map(abs,map(operator.sub, epsilons, epsilon_list[-1]))).expectation())
#    delta_diff.append(lift(map(abs,map(operator.sub, deltas, delta_list[-1]))).expectation())

#if e.validation:
#
#    fig, ax1 = plt.subplots()
#    ax1.plot(ratio, epsilon_diff, 'b-')
#    ax1.set_xlabel('test / entire dataset ratio')
#    ax1.set_ylabel(r'mean of $\Delta\epsilon$/$\epsilon_0$')
#
#    ax2 = ax1.twinx()
#    ax2.plot(ratio, delta_diff, 'r-')
#    ax2.set_ylabel(r'mean of $\Delta\delta$/$\delta_0$')

if e.save_figure is not None:
    plt.savefig(e.save_figure)

if e.show_figure:
    plt.show()

#if __name__ == "__main__":
#    main()
