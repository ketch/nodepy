import matplotlib as mpl
import pylab as pl
import rooted_trees as rt
mpl.use('PS')
fig_width_pt=546.0
inches_per_pt = 1.0/72.27
golden_mean = (pl.sqrt(5)-1.)/2.
fig_width = 9
fig_height = 6
fig_size = [fig_width,fig_height]
params = {'axes.labelsize': 10,
          'text.fontsize':  30,
          'title.fontsize':  30,
          'font.size':  20,
          'legend.fontsize': 10,
          'xtick.labelsize': 8,
          'ytick.labelsize': 8,
          'text.usetex': False,
          'backend': 'ps',
          'figure.figsize': fig_size}

mpl.rcParams.update(params)
rt.plot_all_trees(6)
mpl.rcParams.update(params)
pl.savefig('fig1.eps')
