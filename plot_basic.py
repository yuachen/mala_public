# basic setting for plots in matplotlib
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
matplotlib.rc('text', usetex=True)

label_size = 30

params = {'figure.figsize': (8.0, 6.0),
          'lines.linewidth': 2.,
          'legend.fontsize': 25,
          'axes.labelsize': label_size,
          'xtick.labelsize': label_size,
          'ytick.labelsize': label_size,
          'xtick.major.pad': 8,
          'ytick.major.pad': 8}

plt.rcParams.update(params)
