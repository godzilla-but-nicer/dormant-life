import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from copy import deepcopy
from casim.Totalistic2D import Totalistic2D, DormantLife

# we'll start with a deterministic model where cells go dormant by exposure
thresh = [[[0, 1, 2, 4, 5, 6, 7, 8], [], [3]],
          [[0], [8, 1, 4, 5, 6, 7], [2, 3]],
          [[0, 4, 5, 6, 7, 8], [1], [2, 3]]]

# we will use this for when we generate the random tables
gol_table = [[[0, 1, 2, 4, 5, 6, 7, 8], [], [3]],
             [[], [], []],
             [[0, 1, 4, 5, 6, 7, 8], [], [2, 3]]]

# parameters
null_models = 1000
trials = 10
time_steps = 100
grid = 50
states = 3
outputs = np.zeros((null_models, trials, grid, grid))

for nm in range(null_models):
    # this is just for setting up the transition table
    # we only change the transitions _from_ dormancy
    null_thresh = deepcopy(thresh)
    transitions = np.random.choice(3, size=8)
    for i in range(states):
        null_thresh[1][i] = list(np.where(transitions == i)[0])
    for t in range(trials):
        model = Totalistic2D(states, thresholds=null_thresh)
        nm_out = np.random.choice([0, 2], size=(grid, grid))
        nm_out = model.simulate(nm_out, time_steps)[-1]
        outputs[nm, t, :, :] = nm_out

# the actual SporeLife model
sl_out = np.zeros((trials, time_steps+1, grid, grid))
for t in range(trials):
    sl = Totalistic2D(states, thresholds=thresh)
    run_out = np.random.choice([0, 2], size=(grid, grid))
    run_out = sl.simulate(run_out, time_steps)
    sl_out[t] = run_out

# Game of Life for additional context
gol_out = np.zeros((trials, time_steps+1, grid, grid))
for t in range(trials):
    gol = Totalistic2D(states, thresholds=gol_table)
    run_out = np.random.choice([0, 2], size=(grid, grid))
    run_out = gol.simulate(run_out, time_steps)
    gol_out[t] = run_out

# original DormantLife for even more context
dl_out = np.zeros((trials, time_steps+1, grid, grid))
for t in range(trials):
    dl = DormantLife()
    run_out = np.random.choice([0, 2], size=(grid, grid))
    run_out = dl.simulate(run_out, time_steps)
    dl_out[t] = run_out


# plot the average over trials for each null model
fraction_living = (outputs == 2).mean(axis=(2, 3))
trial_average = fraction_living.mean(axis=1)
# convert to dataframe for seaborn plotting

fig, ax = plt.subplots(ncols=2, sharey=True, gridspec_kw={'width_ratios': [3, 1]})
# now for the real sporelife model on top
sl_living = (sl_out == 2).mean(axis=(2, 3))
sl_average = sl_living.mean(axis=0)

# DormantLife Model as well
dl_living = (dl_out == 2).mean(axis=(2, 3))
dl_average = dl_living.mean(axis=0)

# finally game of life model on top
gol_living = (gol_out == 2).mean(axis=(2, 3))
gol_average = gol_living.mean(axis=0)
# do it
ax[0].plot(np.arange(time_steps+1), gol_average, c='C0', label='Game of Life')
ax[0].plot(np.arange(time_steps+1), dl_average, c='C1', label='DormantLife', linestyle='--')
#ax[0].plot(np.arange(time_steps+1), sl_average, c='C2', label='Spore Life')

ax[0].legend()
ax[0].set_xlabel('Time (step)')
ax[0].set_ylabel('Fraction Living Cells')

# the right hand panel is a histogram of nullmodel outcomes
#ax[1].axhline(sl_average[-1], c='C2')
ax[1].axhline(dl_average[-1], c='C1', linestyle='--')
ax[1].axhline(gol_average[-1], c='C0')
ax[1].hist(trial_average, orientation='horizontal',
           bins=np.arange(0, 0.31, 0.01), edgecolor='black', facecolor='Grey')
ax[1].set_xlabel('Null Models')

ax[0].set_ylim(0.0, 0.33)
ax[1].set_xticks((0, 50, 100))

plt.tight_layout()
plt.savefig('plots/null_three_states_no_null_nosl.pdf')
plt.savefig('plots/null_three_states_no_null_nosl.png')
plt.show()