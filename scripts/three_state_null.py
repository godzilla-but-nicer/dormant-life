import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from copy import deepcopy
from casim.Totalistic2D import Totalistic2D, DormantLife

# we'll start with a deterministic model where cells go dormant by exposure
thresh = [[[0, 1, 2, 4, 5, 6, 7, 8], [], [3]],
          [[8], [0, 1, 4, 5, 6, 7], [2, 3]],
          [[0, 4, 5, 6, 7, 8], [1], [2, 3]]]

# we will use this for when we generate the random tables
gol_table = [[[0, 1, 2, 4, 5, 6, 7, 8], [], [3]],
             [[], [], []],
             [[0, 1, 4, 5, 6, 7, 8], [], [2, 3]]]

# parameters
null_models = 1000
trials = 10
time_steps = 100
grid = 100
states = 3
save_every = 20
saves = int(time_steps / save_every)
outputs = np.zeros((null_models, trials, saves, grid, grid))

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
        for s in range(saves):
            nm_out = model.simulate(nm_out, save_every)[-1]
            outputs[nm, t, s, :, :] = nm_out

# the actual SporeLife model
sl_out = np.zeros((trials, saves, grid, grid))
for t in range(trials):
    sl = Totalistic2D(states, thresholds=thresh)
    run_out = np.random.choice([0, 2], size=(grid, grid))
    for s in range(saves):
        run_out = sl.simulate(run_out, save_every)[-1]
        sl_out[t, s, :, :] = run_out

# Game of Life for additional context
gol_out = np.zeros((trials, saves, grid, grid))
for t in range(trials):
    gol = Totalistic2D(states, thresholds=gol_table)
    run_out = np.random.choice([0, 2], size=(grid, grid))
    for s in range(saves):
        run_out = gol.simulate(run_out, save_every)[-1]
        gol_out[t, s, :, :] = run_out

# original DormantLife for even more context
dl_out = np.zeros((trials, saves, grid, grid))
for t in range(trials):
    dl = DormantLife()
    run_out = np.random.choice([0, 2], size=(grid, grid))
    for s in range(saves):
        run_out = dl.simulate(run_out, save_every)[-1]
        dl_out[t, s, :, :] = run_out

# plot the average over trials for each null model
fraction_living = (outputs == 2).mean(axis=(3, 4))
trial_average = fraction_living.mean(axis=1)
# convert to dataframe for seaborn plotting
col_nums = np.arange(1, saves+1)*save_every
col_names = ['t' + str(num) for num in col_nums]
null_df = pd.DataFrame(trial_average, columns=col_names)
null_df['id'] = np.arange(null_df.shape[0])
null_df_long = (pd.wide_to_long(null_df, stubnames='t', i='id', j='time')
                .reset_index())

fig, ax = plt.subplots(ncols=2, sharey=True, gridspec_kw={'width_ratios': [3, 1]})
sns.histplot(x='time', y='t', data=null_df_long, cmap='Greys', ax=ax[0],
             binwidth=(save_every, 0.01),
             binrange=((save_every, time_steps), (0, 0.3)))

# now for the real sporelife model on top
sl_living = (sl_out == 2).mean(axis=(2, 3))
sl_average = sl_living.mean(axis=0)
ax[0].plot(np.arange(1, 1+saves)*save_every, sl_average, c='C2', label='Spore Life')

# DormantLife Model as well
dl_living = (dl_out == 2).mean(axis=(2, 3))
dl_average = dl_living.mean(axis=0)
ax[0].plot(np.arange(1, 1+saves)*save_every, dl_average, c='C1', label='DormantLife')

# finally game of life model on top
gol_living = (gol_out == 2).mean(axis=(2, 3))
gol_average = gol_living.mean(axis=0)
ax[0].plot(np.arange(1, 1+saves)*save_every, gol_average, c='C0', label='Game of Life')

ax[0].legend()
ax[0].set_xlabel('Time (step)')
ax[0].set_ylabel('Fraction Living Cells')

# the right hand panel is a histogram of nullmodel outcomes
ax[1].axhline(sl_average[-1], c='C2')
ax[1].axhline(dl_average[-1], c='C1')
ax[1].axhline(gol_average[-1], c='C0')
ax[1].hist(trial_average[:, -1], orientation='horizontal',
           bins=np.arange(0, 0.31, 0.01), edgecolor='black', facecolor='Grey')
ax[1].set_xlabel('Null Models')

plt.tight_layout()
plt.savefig('../plots/null_three_states.pdf')
plt.savefig('../plots/null_three_states.png')
plt.show()
