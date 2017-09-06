import matplotlib
import matplotlib.pyplot as plt

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 10}

matplotlib.rc('font', **font)

plt.rc('text', usetex=True)
from matplotlib.pylab import rcParams
#rcParams['hatch.linewidth'] = 3
rcParams['figure.figsize'] = 4.0, 2.0

#rcParams['figure.figsize'] = 8.0, 4.0
