#!/bin/python
import numpy as np
import seaborn as sns
import os
import pandas as pd
import time
import casim.Totalistic2D as Tot2D
import matplotlib.pyplot as plt
from scipy import stats

# set up parameters
grid_size = 7
trials = 20000
max_steps = 30000

# game of life
gol_start = time.time()
os.system(f"./carust -f rule_tables/game_of_life.txt --transient {grid_size} {max_steps} {trials}")
gol_tlist = []
with open(f"data/gol_transients_{grid_size}_{max_steps}_{trials}.csv", 'r') as tin:
    data = tin.read().strip().split(',')
    for val in data:
        if val != '':
            gol_tlist.append(int(val))
gol_trans = np.array(gol_tlist)

gol_time = time.time() - gol_start
print(f"Game of Life time: {gol_time:.3f} s")

# spore life
sl_start = time.time()
os.system(f"./carust -f rule_tables/spore_life.txt --transient {grid_size} {max_steps} {trials}")
sl_tlist = []
with open(f"data/sl_transients_{grid_size}_{max_steps}_{trials}.csv", 'r') as tin:
    data = tin.read().strip().split(',')
    for val in data:
        if val != '':
            sl_tlist.append(int(val))
sl_trans = np.array(sl_tlist)
sl_time = time.time() - sl_start
print(f"Spore Life time: {sl_time:.3f} s")

all_transients = np.hstack((gol_trans, sl_trans))
labels = np.hstack((['Game of Life']*trials, ['Spore Life']*trials))

df = pd.DataFrame({'Model': labels, 'Transient Length': all_transients})

sns.histplot(x='Transient Length', hue='Model', data=df, stat='count', element='bars', log_scale=True)
plt.axvline(np.mean(df[df['Model'] == 'DormantLife']['Transient Length']), c='C1', lw=5, linestyle='--')
plt.axvline(np.mean(df[df['Model'] == 'Game of Life']['Transient Length']), c='C0', lw=5, linestyle='--')
plt.savefig('transient_dists_77.pdf')
plt.savefig('transient_dists_77.png')
plt.show()

df.to_csv('transients_77.csv')
