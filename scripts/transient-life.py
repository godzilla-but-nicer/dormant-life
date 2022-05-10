import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy as spentropy
from casim.Totalistic2D import GameOfLife, DormantLife
from matplotlib import animation

# keep this so we can use it more than once
initial_grid = np.random.choice([0, 1, 2], size=[6, 6], p=[0.34, 0.33, 0.33])

gol = DormantLife(noise=0.0,seed=420)

# get the transients and the grid at each step
hist, trans = gol.simulate_transients(initial_grid, 1000)
more_hist = gol.simulate(hist[-1], hist.shape[0])[1:]
long_hist = np.vstack((hist, more_hist))

print(trans)
print(hist.shape)
print(more_hist.shape)
print(long_hist.shape)

fig, ax = plt.subplots(figsize=(8, 8))
ax.axis('off')
im = ax.imshow(long_hist[0], interpolation='none', cmap='Blues')

title_text = 'Step: {0:d}. H={1:.2f}'


def draw_life(i):
    if i == trans+1:
        im.set(cmap = 'Reds')

    im.set_array(long_hist[i])
    
    prob = np.sum(long_hist[i]) / (long_hist[i].shape[0]**2)
    entropy = spentropy([prob, 1-prob], base=2)
    
    ax.set_title(title_text.format(i, entropy))
    return im

ani = animation.FuncAnimation(fig, draw_life, interval=80, frames=long_hist.shape[0])
# ani.save('dlife.gif', writer='imagemagick')
ani.save('plots/transient.mp4', writer='ffmpeg')