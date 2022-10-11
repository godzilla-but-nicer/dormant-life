#%%
import numpy as np
import matplotlib.pyplot as plt
import casim.Totalistic2D as ct
from matplotlib.colors import ListedColormap
from scipy.signal import convolve2d

# load the rule tables
sl_rules = np.loadtxt('../rule_tables/spore_life.txt', delimiter=',')
print(sl_rules)
sl = ct.Totalistic2D(3, sl_rules)

#%%
gol_init = np.array([[1, 0, 0],
                     [0, 0, 0],
                     [0, 0, 1]])

sl_init = gol_init.copy()

for i in range(sl_init.shape[0]):
    for j in range(sl_init.shape[1]):
        if gol_init[i, j] == 0:
            if np.random.uniform() < 0.20:
                sl_init[i, j] = 1
        elif gol_init[i, j] == 1:
            sl_init[i, j] = 2


sl_seq = sl.simulate(sl_init, 4)
sl_colors = [(1, 1, 1), (255/255, (191+19)/255, (141+14)/255), (1., 127/255, 14/255)]
sl_cmap = ListedColormap(sl_colors, N=3)

color_array = np.array(sl_colors)
def grid_to_colors(grid, colors):
    if len(grid.shape) == 2:
        mapped_colors = np.zeros((grid.shape[0], grid.shape[1], 3))
        for i in range(mapped_colors.shape[0]):
            for j in range(mapped_colors.shape[1]):
                mapped_colors[i, j, :] = colors[int(grid[i, j])]
    elif len(grid.shape) == 1:
        mapped_colors = np.zeros((grid.shape[0], 1, 3))
        for i in range(mapped_colors.shape[0]):
            mapped_colors[i, 0, :] = colors[int(grid[i])]
    
    return mapped_colors


print(sl_seq)

for ci, con in enumerate(sl_seq):
    fig, ax = plt.subplots(figsize=(3,3))
    ax.imshow(grid_to_colors(con, color_array), cmap=sl_cmap)
    ax.grid(which='minor', linewidth=2, c='k')
    ax.set_xticks(np.arange(-1, 3)+0.5, minor=True)
    ax.set_yticks(np.arange(-1, 3)+0.5, minor=True)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(f"../plots/explanatory_graphic/sl_stg_{ci}.pdf")
#%%

for ci, con in enumerate(sl_seq):
    fig, ax = plt.subplots(figsize=(3,1))

    flat = con.flatten()
    row = flat.reshape(1, flat.shape[0])

    ax.imshow(grid_to_colors(row, color_array), cmap=sl_cmap)
    ax.grid(which='minor', linewidth=2, c='k')
    ax.set_xticks(np.arange(-1, 9)+0.5, minor=True)
    ax.set_yticks(np.arange(-1, 1)+0.5, minor=True)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(f"../plots/explanatory_graphic/sl_stg_flat_{ci}.pdf")



# %%
import casim.utils as cu
b3 = cu.to_base(16373, 3, 9)
b10 = cu.to_decimal(b3, 9, 3)
# %%
