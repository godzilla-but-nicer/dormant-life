import numpy as np
import matplotlib.pyplot as plt
from casim.Totalistic2D import GameOfLife, DormantLife
from matplotlib import animation

initial_conditions = np.random.choice((0, 1, 2), p=(0.25, 0.25, 0.5), size=[20, 20])
# initial_conditions = np.zeros((100, 100))
# initial_conditions[20:80, 20:80] = 2
gol_cond = initial_conditions == 2

n_steps = 500
step_range = np.arange(n_steps)

# DormantLife simulation and alive fraction
gol = DormantLife(420)
jist = gol.simulate(initial_conditions, n_steps)
dl_alive = [np.sum(h == 2) / h.shape[0]**2 for h in jist]

# initialize Game of Life
gol = GameOfLife(420)
hist = gol.simulate(gol_cond, n_steps)
gol_alive = [np.sum(h == 1) / h.shape[0]**2 for h in hist]

# initialize basic plot objects
fig = plt.figure(constrained_layout=True, figsize=(8, 8))
ax = fig.subplot_mosaic([['A', 'A'],
                         ['B', 'C']])

# initialize the empty line plot
gol_lines, = ax['A'].plot(step_range[0], gol_alive[0], label='Game of Life')
gol_pt = ax['A'].scatter(step_range[0], gol_alive[0])
dl_lines, = ax['A'].plot(step_range[0], dl_alive[0], label='DormantLife')
dl_pt = ax['A'].scatter(step_range[0], dl_alive[0])

# limits and labels
ax['A'].set_xlim(0, n_steps)
ax['A'].set_ylim(-0.01, 1)
ax['A'].legend()
ax['A'].set_xlabel('Time Steps')
ax['A'].set_ylabel('Fraction Living Cells')

# initialize empty images
ax['B'].axis('off')
ax['C'].axis('off')
im = ax['B'].imshow(hist[0], interpolation='none', cmap='Blues')
jim = ax['C'].imshow(jist[0], interpolation='none', cmap='Oranges')

ax['B'].set_title('Game Of Life')
ax['C'].set_title('Dormant Life')


def draw_life(i):
    gol_lines.set_data(step_range[:i], gol_alive[:i])
    gol_pt.set_offsets((step_range[i], gol_alive[i]))
    dl_lines.set_data(step_range[:i], dl_alive[:i])
    dl_pt.set_offsets((step_range[i], dl_alive[i]))
    im.set_array(hist[i])
    jim.set_array(jist[i])
    return im


ani = animation.FuncAnimation(fig, draw_life, interval=100, frames=n_steps-1)
ani.save('dlife.mp4', writer='ffmpeg')
#ani.save('dlife.gif', writer='imagemagick')




