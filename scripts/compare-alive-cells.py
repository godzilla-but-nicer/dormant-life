import numpy as np
import matplotlib.pyplot as plt
from casim.Totalistic2D import DormantLife, GameOfLife

# params
n_inits = 100
m_steps = 50
grid_size = 10
bootstrap_num = 10000
low_percentile = 5
high_percentile = 95

# initialize both models
gol = GameOfLife(123)
dl = DormantLife(123)

# hard to compare directly so we will just simulate a bunch of random starts
# and get distributions of number of alive cells.
# I think we want to just count frome like a few steps after the random init
gol_alive = np.zeros((n_inits, m_steps))
dl_alive = np.zeros((n_inits, m_steps))

for i in range(n_inits):
    gol_init = np.random.choice(2, size=(grid_size, grid_size))
    dl_init = np.random.choice([0, 1, 2], p=[0.25, 0.25, 0.5], size=(grid_size, grid_size))

    gol_hist = gol.simulate(gol_init, m_steps + 5)
    dl_hist = dl.simulate(dl_init, m_steps + 5)

    for j in range(m_steps):
        gol_alive[i, j] = np.sum(gol_hist[j] == 1)
        dl_alive[i, j] = np.sum(dl_hist[j] == 2)

# Bootstrap each step to get confidence intervals
dl_boots_low = []
dl_boots_high = []
gol_boots_low = []
gol_boots_high = []

for j in range(m_steps):
    gol_boots_means = []
    dl_boots_means = []

    for k in range(bootstrap_num):
        bootstrap_gol = np.random.choice(gol_alive[:, j], size=n_inits, replace=True)
        gol_boots_means.append(np.mean(bootstrap_gol))

        bootstrap_dl = np.random.choice(dl_alive[:, j], size=n_inits, replace=True)
        dl_boots_means.append(np.mean(bootstrap_dl))

    # outside of the resampling loop we can calculate the percentiles
    dl_boots_low.append(np.percentile(dl_boots_means, low_percentile))
    dl_boots_high.append(np.percentile(dl_boots_means, high_percentile))
    gol_boots_low.append(np.percentile(gol_boots_means, low_percentile))
    gol_boots_high.append(np.percentile(gol_boots_means, high_percentile))

gol_means = np.mean(gol_alive, axis=0)
gol_ses = np.std(gol_alive, axis=0) #/ np.sqrt(n_inits)
dl_means = np.mean(dl_alive, axis=0)
dl_ses = np.std(dl_alive, axis=0) #/ np.sqrt(n_inits)

print(gol_boots_low)
print(gol_means)
print(gol_boots_high)
# gol line with error
plt.plot(gol_means, label='Game of Life', c='C0', marker='o', ms=3)
plt.fill_between(range(m_steps), gol_means, gol_boots_high, 
                 linestyle='--', color='C0', alpha=0.4)
plt.fill_between(range(m_steps), gol_boots_low, gol_means, 
                 linestyle='--', color='C0', alpha=0.4)

# dormant line with error
plt.plot(dl_means, label='DormantLife', c='C1', marker='s', ms=3)
plt.fill_between(range(m_steps), dl_means, dl_boots_high, 
                 linestyle='--', color='C1', alpha=0.4)
plt.fill_between(range(m_steps), dl_boots_low, dl_means, 
                 linestyle='--', color='C1', alpha=0.4)

plt.legend()
plt.xlabel('Time [steps]')
plt.ylabel(r'Number Living Cells $\pm$ 95% bootstrap')
plt.savefig('livingcells.png')
plt.savefig('livingcells.pdf')
plt.show()