# coding=utf-8

import sys
reload(sys)
sys.setdefaultencoding('utf8')
sys.tracebacklimit=2

from detect  import all_stats,Decomposition
from util    import *
from ml_util import *
from math    import log
import sklearn.cross_validation as cross_validation

from plot_util import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, LogLocator, FixedLocator, NullFormatter
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
from scipy.interpolate import spline

gen = generated_from_args()
metric = "nmi"

color_map = {'red':   (0.1,0.1,0.1,1.0),
             'blue':  (0.3,0.3,0.3,1.0),
             'green': (0.6,0.6,0.6,1.0),
             'black': (0.0,0.0,0.0,1.0)}

width_map = {'red'  : 0.25,
             'blue' : 1.25,
             'green': 1.25,
             'black': 0.25}

marker_map = {'red':   '*',
              'green': '+',
              'blue':  'o',
              'black': 'o',}

def actual_color(c):
    global gen
    
    if gen.args.bw:
        if c in color_map:
            return color_map[c]
        else:
            return 'black'
    else:
        return c

# Lists used in storing epsilon and delta values
epsilons_list  = []      # epsilon
deltas_list    = []      # delta
sizes_list     = []      # size
arrows_list    = []
colors_list    = []
labels_list    = []
models_list    = []
idents_list    = []

annots_list    = []

for data in gen.data1:

    epsilon_list = []      # epsilon
    delta_list   = []      # delta
    size_list    = []      # size
    color_list   = []
    label_list   = []
    model_list   = []
    ident_list   = []
    
    for row_ in data.iterrows():
        row = row_[1]

        if row['epsilon'] != 0.0 and row['delta'] != 0.0:
            epsilon_list.append(row['epsilon'])
            delta_list  .append(row['delta'])
            size_list   .append(5*log(row['size'],1.5))
            color_list  .append(row['color'])
            label_list  .append(row['label'])
            model_list  .append(row['model'])
            ident_list  .append(row['ident'])


    epsilons_list.append(epsilon_list)
    deltas_list  .append(delta_list)
    sizes_list   .append(size_list)
    colors_list  .append(color_list)
    labels_list  .append(label_list)
    models_list  .append(model_list)
    idents_list  .append(ident_list)

for data in gen.data2:
    arrow_list = []
    
    for row_ in data.iterrows():
        row = row_[1]
        #if row['epsilon'] != 0.0 and row['delta'] != 0.0:
        arrow_list.append({'parent':(row['parent_delta'],row['parent_epsilon']),
                           'child': (row['child_delta'], row['child_epsilon']),
                           'color': row['color']})

    arrows_list.append(arrow_list)

for data in gen.data3:
    epsilon_list  = []      # epsilon
    delta_list    = []      # delta
    size_list     = []      # size
    color_list    = []
    label_list    = []
    model_list    = []
    ident_list    = []

    annot_list    = []
    
    for row_ in data.iterrows():
        row = row_[1]

        #print row

        label = {'p': (row['delta'], row['epsilon']),
                 'label': row['label'],
                 'color': row['color'],
                 'va': row['va'],
                 'ha': row['ha'],
                 'xytext': (row['xtext'], row['ytext'])}

        annot_list.append(label)
    
        if row['epsilon'] != 0.0:
            epsilon_list.append(row['epsilon'])
            delta_list  .append(row['delta'])
            size_list   .append(0.0)
            color_list  .append(row['color'])
            label_list  .append(row['label'])
            model_list  .append(row['model'])
            ident_list  .append(row['ident'])
            #annot_list  .append(row['label'])

    annots_list.append(annot_list)

    epsilons_list.append(epsilon_list)
    deltas_list  .append(delta_list)
    sizes_list   .append(size_list)
    colors_list  .append(color_list)
    labels_list  .append(label_list)
    models_list  .append(model_list)
    idents_list  .append(ident_list)

xmin = 2**(log(max(0.000000001, min(lists_flatten(deltas_list))),   2)-0.5)
ymin = 2**(log(max(0.000000001, min(lists_flatten(epsilons_list))), 2)-0.5)

#print epsilon_list

xmax = 2**(log(max(lists_flatten(deltas_list)), 2)+0.5)
ymax = 2**(log(max(lists_flatten(epsilons_list)), 2)+0.5)

#plt.autoscale(enable=True,tight=False,axis='both')

ax = plt.axes()
ax.set_yscale('log', basey=2)
ax.set_xscale('log', basex=2)

plt.ylim([ymin,2**(0.5)])
plt.xlim([xmin,xmax])

# plot labels
plt.xlabel(r'$\delta$ / influence [probability]')
plt.ylabel(r'$\epsilon$ / association (%s)' % (metric))
plt.grid(b=True, linewidth=0.1, linestyle=':')

bbox_props=dict(boxstyle="round,pad=0.3", fc="white", lw=0, alpha=0.75)

handles = []

legend_added = {}

if True:
    for arrow_list in arrows_list:
        for a in arrow_list:

            p1    = a['parent']
            p2    = a['child']
            color = a['color']
            linewidth = width_map[color]
            q = plt.quiver(p1[0],p1[1],
                           p2[0]-p1[0],p2[1]-p1[1],
                           units='dots', width=width_map[color],
                           scale_units='xy', angles='xy',
                           scale=1,
                           color=actual_color(color),
                           alpha=0.50,
                           zorder=10,
                           )

allow_legends = set(['lasso','logistic','random-forest','decision-tree'])

for (epsilon_list,delta_list,size_list,color_list,label_list,model_list,ident_list) in zip(epsilons_list,deltas_list,sizes_list,colors_list,labels_list,models_list,idents_list):
    marker  = marker_map[color_list[0]]
    width   = width_map[color_list [0]]
    mcolors = map(actual_color, color_list)
    
    label = model_list[0]
    ident = ident_list[0]
    model = model_list[0]

    if ident in legend_added:
        ident = None
    else:
        legend_added[ident] = True

    if model not in allow_legends:
        ident = None
        
    plt.scatter(delta_list, epsilon_list,
                marker=marker, sizes=size_list,
                edgecolors=mcolors, facecolors=mcolors,
                zorder=10, linewidth=1.0)

    plt.scatter([], [], marker=marker, label=ident, color=mcolors[0])

for annot_list in annots_list:
    for label in annot_list:
        if label['label'] == 'A':
            marker = marker_map[label['color']]
            ax.add_patch(
                patches.Rectangle((0.0,0.0),
                                  label['p'][0],
                                  label['p'][1],
                                  facecolor=actual_color(label['color']),
                                  alpha=0.10
                                  )
                )

        elif label['label'] == 'threshold':
            ax.add_patch(
                patches.Rectangle((label['p'][0],label['p'][1]),
                                  10.0,10.0,
                                  edgecolor='none',
                                  facecolor=(0.0,0.0,0.0,0.5),
                                  linewidth=1.0,
                                  zorder=-10
                                  )
                )
        else:
            if label['p'][1] > gen.args.epsilon and label['p'][0] > gen.args.delta:
                ax.annotate(label['label']  ,color  = actual_color(label['color']),
                            xy = label['p'] ,xytext = label['xytext'],
                            va = label['va'],ha = label['ha'],
                            xycoords='data',
                            textcoords='offset points',
                            bbox=bbox_props,
                            arrowprops=dict(linewidth=0.25,
                                            linestyle=':',
                                            arrowstyle='-',
                                            shrinkA=0.0,
                                            shrinkB=0.0,
                                            facecolor=actual_color(label['color']),
                                            edgecolor=actual_color(label['color']),
                                            )
                            )

xmajorLocator = FixedLocator(locs = [2.0**(-i) for i in range(0,20,2)])
ymajorLocator = FixedLocator(locs = [2.0**(-i) for i in range(0,20,2)])

tics = 5
subs = [1.0+3.0*(float(i)/float(tics)) for i in range(1,tics+1)]

xminorLocator = FixedLocator(locs = [2.0**(-i) * sub for sub in subs for i in range(0,20,2)])
yminorLocator = FixedLocator(locs = [2.0**(-i) * sub for sub in subs for i in range(0,20,2)])

plt.axes().xaxis.set_minor_formatter(NullFormatter())
plt.axes().yaxis.set_minor_formatter(NullFormatter())
plt.axes().yaxis.set_minor_locator(yminorLocator)
plt.axes().xaxis.set_minor_locator(xminorLocator)
plt.axes().yaxis.set_major_locator(ymajorLocator)
plt.axes().xaxis.set_major_locator(xmajorLocator)

plt.axes().yaxis.set_tick_params(which='major', right = 'off', left   ='on', labelbottom=True, labeltop=False, labelleft=True, labelright=False)
plt.axes().xaxis.set_tick_params(which='major', top   = 'off', bottom ='on', labelbottom=True, labeltop=False, labelleft=True, labelright=False)
plt.axes().yaxis.set_tick_params(which='minor', right = 'off', left   ='on', labelbottom=False, labeltop=False, labelleft=False, labelright=False)
plt.axes().xaxis.set_tick_params(which='minor', top   = 'off', bottom ='on', labelbottom=False, labeltop=False, labelleft=False, labelright=False)

#plt.legend(handles=handles,loc='upper left',handlelength=1, frameon=True)
plt.legend(loc='upper left',handlelength=1, frameon=True)

plt.tight_layout()

if gen.args.output is not None:
    plt.savefig(gen.args.output)

if gen.args.show:
    plt.show()


