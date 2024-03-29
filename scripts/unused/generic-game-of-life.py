import numpy as np
import matplotlib.pyplot as plt
from casim.Totalistic2D import Totalistic2D
from matplotlib import animation

# glider = np.array([[0, 2, 0],
#                    [0, 0, 2],
#                    [2, 2, 2]])
thresh  = [[[0, 1, 2, 4, 5, 6, 7, 8], [], [3]],
          [[8], [0, 1, 4, 5, 6, 7], [2, 3]],
          [[0, 4, 5, 6, 7, 8], [1], [2, 3]]]
gol = Totalistic2D(n_states=3, thresholds=thresh)
for _ in range(100):
    initial_conditions = np.random.choice([0, 2], p=[0.50,  0.5], size=[7, 7])
    t_len, hist = gol.simulate_transients(initial_conditions, 10000)
# 
# fig, ax = plt.subplots(figsize=(8,4))
# ax.axis('off')
# im = ax.imshow(hist[0], interpolation='none', cmap='Greys')
# 
# ax.set_title('SporeLife')
# 
# def draw_life(i):
#     im.set_array(hist[i])
#     return im
# 
# ani = animation.FuncAnimation(fig, draw_life, interval=100, frames=hist.shape[0])
# ani.save('plots/life.mp4', writer='ffmpeg')
# 