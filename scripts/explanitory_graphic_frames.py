#%%
import numpy as np
import matplotlib.pyplot as plt
import casim.Totalistic2D as ct
from matplotlib.colors import ListedColormap
from scipy.signal import convolve2d

# load the rule tables
gol_rules = np.loadtxt('../rule_tables/game_of_life.txt', delimiter=',')
sl_rules = np.loadtxt('../rule_tables/spore_life.txt', delimiter=',')

gol = ct.GameOfLife()
sl = ct.Totalistic2D(3, sl_rules)
# %%
gol_rand_init = np.random.choice(2, replace=True, size=(10, 10))
gol_init = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                     [0, 1, 0, 1, 0, 0, 1, 1, 1, 0],
                     [0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
                     [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

gol_seq = gol.simulate(gol_init, 6)

gol_colors = [(1, 1, 1), (0.12157, 0.46667, 0.70588)]
gol_cmap = ListedColormap(gol_colors)

for ci, con in enumerate(gol_seq):
    fig, ax = plt.subplots(figsize=(3,3))
    ax.imshow(con, cmap=gol_cmap)
    ax.grid(which='minor', linewidth=2, c='k')
    ax.set_xticks(np.arange(-1, 10)+0.5, minor=True)
    ax.set_yticks(np.arange(-1, 10)+0.5, minor=True)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(f"../plots/explanatory_graphic/gol_{ci}.pdf")

#%%
c_filter = np.array(([1, 1, 1], [1, 0, 1], [1, 1, 1]))
for ci, con in enumerate(gol_seq):
    neighbors = convolve2d(con, c_filter, mode='same', boundary='wrap')
    fig, ax = plt.subplots(figsize=(3,3))
    ax.imshow(con, cmap=gol_cmap)
    ax.grid(which='minor', linewidth=2, c='k')
    ax.set_xticks(np.arange(-1, 10)+0.5, minor=True)
    ax.set_yticks(np.arange(-1, 10)+0.5, minor=True)
    ax.set_xticks([])
    ax.set_yticks([])
    for i in range(con.shape[0]):
        for j in range(con.shape[1]):
            ax.text(j, i, s=f"{int(neighbors[i, j]):d}", ha='center', va='center')
    plt.savefig(f"../plots/explanatory_graphic/gol_neighbors_{ci}.pdf")
    plt.show()
# %%
fig, ax = plt.subplots(figsize=(6.5, 2))
ax.imshow(gol_rules[:, :8], cmap=gol_cmap)
ax.set_yticks((0, 1))
ax.set_yticklabels(("DEAD", "ALIVE"))
ax.grid(which='minor', linewidth=2, c='k')
ax.set_xticks(np.arange(-1, 8)+0.5, minor=True)
ax.set_yticks(np.arange(-1, 2)+0.5, minor=True)
ax.set_xlabel("Number of ALIVE Neighbors")
ax.set_ylabel("Current State")

for i in range(2):
    for j in range(8):
        if gol_rules[i, j] == 1:
            next_state = "ALIVE"
        else:
            next_state = "DEAD"

        ax.text(j, i, next_state, ha='center', va='center')

plt.tight_layout()
plt.savefig("../plots/explanatory_graphic/gol_rules.pdf")

#%%
sl_init = gol_init.copy()

for i in range(sl_init.shape[0]):
    for j in range(sl_init.shape[1]):
        if gol_init[i, j] == 0:
            if np.random.uniform() < 0.333:
                sl_init[i, j] = 1
        elif gol_init[i, j] == 1:
            sl_init[i, j] = 2


sl_seq = sl.simulate(sl_init, 6)
print(sl_seq)

sl_colors = [(1, 1, 1), (255/255, 191/255, 141/255), (1., 127/255, 14/255)]
sl_cmap = ListedColormap(sl_colors, N=3)

for ci, con in enumerate(sl_seq):
    fig, ax = plt.subplots(figsize=(3,3))
    ax.imshow(con, cmap=sl_cmap)
    ax.grid(which='minor', linewidth=2, c='k')
    ax.set_xticks(np.arange(-1, 10)+0.5, minor=True)
    ax.set_yticks(np.arange(-1, 10)+0.5, minor=True)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(f"../plots/explanatory_graphic/sl_{ci}.pdf")
# %%
fig, ax = plt.subplots(figsize=(6.5, 2.5))
ax.imshow(sl_rules, cmap=sl_cmap)
ax.set_yticks((0, 1, 2))
ax.set_yticklabels(("DEAD", "DORM", "ALIVE"))
ax.grid(which='minor', linewidth=2, c='k')
ax.set_xticks(np.arange(-1, 8)+0.5, minor=True)
ax.set_yticks(np.arange(-1, 3)+0.5, minor=True)
ax.set_ylabel("Current State")
ax.set_xlabel("Number of ALIVE Neighbors")
print(sl_rules)

for i in range(3):
    for j in range(8):
        if sl_rules[i, j] == 2:
            next_state = "ALIVE"
        elif sl_rules[i, j] == 1:
            next_state = "DORM"
        else:
            next_state = "DEAD"

        ax.text(j, i, next_state, ha='center', va='center', zorder=100)

plt.tight_layout()
plt.savefig("../plots/explanatory_graphic/sl_rules.pdf")


# %%
