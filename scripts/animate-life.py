import numpy as np
import matplotlib.pyplot as plt
from casim.Totalistic2D import GameOfLife, DormantLife
from matplotlib import animation

initial_conditions = np.random.choice([0, 2], p=[0.90, 0.10], size=[10, 10])
# initial_conditions = np.zeros((100, 100))
# initial_conditions[20:80, 20:80] = 2
gol_cond = initial_conditions == 2

gol = DormantLife(420)
jist = gol.simulate(initial_conditions, 200)

gol = GameOfLife(420)
hist = gol.simulate(gol_cond, 200)

fig, ax = plt.subplots(ncols=2, figsize=(8,4))
ax[0].axis('off')
ax[1].axis('off')
im = ax[0].imshow(hist[0], interpolation='none', cmap='Greys')
jim = ax[1].imshow(jist[0], interpolation='none', cmap='Greys')

ax[0].set_title('Game Of Life')
ax[1].set_title('Dormant Life')


def draw_life(i):
    im.set_array(hist[i])
    jim.set_array(jist[i])
    return im


ani = animation.FuncAnimation(fig, draw_life, interval=100, frames=hist.shape[0])
# ani.save('dlife.gif', writer='imagemagick')
ani.save('dlife.mp4', writer='ffmpeg')



