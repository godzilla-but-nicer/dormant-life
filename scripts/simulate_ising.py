from dormantspin.models import Ising
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

ii = Ising({'J': 1, "h":0}, 100)

vis_states = [ii.grid]
for i in range(100):
    time_series = ii.metropolis(10000, 10)
    vis_states.append(time_series[-1])

fig, ax = plt.subplots()
im = ax.imshow(vis_states[0], interpolation='none', cmap='Greys')
def animate(i):
    im.set_array(vis_states[i])
    return im


ani = FuncAnimation(fig, animate, interval=50, frames=len(vis_states))
ani.save('plots/ising.mp4', 'ffmpeg')
