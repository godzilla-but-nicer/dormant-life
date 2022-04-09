#!/bin/python
import numpy as np
import seaborn as sns
import pandas as pd
import casim.Totalistic2D as Tot2D
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm

# set up parameters
grid_size = 7
n_trials = 10000
max_steps = 10000

# initialize CA simulators
gol = Tot2D.GameOfLife(420)
dl = Tot2D.DormantLife(420)

# arrays for transients
gol_transients = np.zeros(n_trials)
dl_transients = np.zeros(n_trials)

for trial in tqdm(range(n_trials)):
    # get initial grids
    init_gol = np.random.choice(2, size=(grid_size, grid_size))
    init_dl = np.random.choice(3, size=(grid_size, grid_size))

    # find the transients
    gol_hist, gol_trans = gol.simulate_transients(init_gol, max_steps)
    dl_hist, dl_trans = dl.simulate_transients(init_dl, max_steps)

    # add 1 - returned is index of final transient state
    gol_transients[trial] = gol_trans + 1
    dl_transients[trial] = dl_trans + 1


all_transients = np.hstack((gol_transients, dl_transients))
labels = np.hstack((['Game of Life']*n_trials, ['DormantLife']*n_trials))

df = pd.DataFrame({'Model': labels, 'Transient Length': all_transients})
# df = pd.read_csv('transients.csv', index_col=0)

sns.histplot(x='Transient Length', hue='Model', data=df, stat='count', element='bars', log_scale=True)
plt.axvline(np.mean(df[df['Model'] == 'DormantLife']['Transient Length']), c='C1', lw=5, linestyle='--')
plt.axvline(np.mean(df[df['Model'] == 'Game of Life']['Transient Length']), c='C0', lw=5, linestyle='--')
plt.savefig('transient_dists_77.pdf')
plt.savefig('transient_dists_77.png')
plt.show()

df.to_csv('transients_77.csv')
