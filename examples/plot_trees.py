import nodepy.rooted_trees as rt
import pylab as pl

rt.plot_all_trees(6)
pl.suptitle('Rooted Trees of Order 6',fontsize=20)
pl.suptitle('Titles are tree representations in Butcher notation',fontsize=12,y=0.05)
pl.draw()
