import numpy as np
import matplotlib.pyplot as plt
import sys
from casim.Totalistic2D import DormantLife, GameOfLife
from tqdm import tqdm

# bootstrap parameters
bootstrap_num = int(sys.argv[1])
low_percentile = int(sys.argv[2])
high_percentile = 100 - low_percentile
n_inits = int(sys.argv[3])
m_steps = int(sys.argv[4])
sl_robust = int(sys.argv[5])
gol_robust = int(sys.argv[6])

# load the simulated time_series and pull average fraction alive at each step
with np.load("data/time_series/sl_time_series.npz") as sl_data:
    sl_alive = np.zeros((n_inits, m_steps))
    for i in range(n_inits):
        trial_ts = sl_data[str(i)]
        trial_alive = trial_ts == 2
        sl_alive[i] = np.mean(trial_alive, axis=(1, 2))


with np.load("data/time_series/gol_time_series.npz") as gol_data:
    gol_alive = np.zeros((n_inits, m_steps))
    for i in range(n_inits):
        trial_ts = gol_data[str(i)]
        gol_alive[i] = np.mean(trial_ts, axis=(1, 2))

# Bootstrap each step to get confidence intervals
sl_boots_low = []
sl_boots_high = []
gol_boots_low = []
gol_boots_high = []


print("Running bootstrap...")
for j in tqdm(range(m_steps)):
    gol_boots_means = []
    sl_boots_means = []

    for k in range(bootstrap_num):
        bootstrap_gol = np.random.choice(gol_alive[:, j], size=gol_alive.shape[0], replace=True)
        gol_boots_means.append(np.mean(bootstrap_gol))

        bootstrap_sl = np.random.choice(sl_alive[:, j], size=sl_alive.shape[0], replace=True)
        sl_boots_means.append(np.mean(bootstrap_sl))

    # outside of the resampling loop we can calculate the percentiles
    sl_boots_low.append(np.percentile(sl_boots_means, low_percentile))
    sl_boots_high.append(np.percentile(sl_boots_means, high_percentile))
    gol_boots_low.append(np.percentile(gol_boots_means, low_percentile))
    gol_boots_high.append(np.percentile(gol_boots_means, high_percentile))

gol_means = np.mean(gol_alive, axis=0)
sl_means = np.mean(sl_alive, axis=0)

# robustness
print("Loading Spore Life robustness rules...")
sl_r_stack = []
checked = set([])
for sl_r in tqdm(range(sl_robust)):

    # do the same proceedure as with the real rule and add it to a list
    with np.load(f"data/time_series/sl_robust_{sl_r}_time_series.npz") as sl_r_data:
        sl_r_alive = np.zeros((n_inits, m_steps))
        for i in range(n_inits):
            trial_ts = sl_r_data[str(i)]
            trial_alive = trial_ts == 2
            sl_r_alive[i] = np.mean(trial_alive, axis=(1, 2))
        sl_r_stack.append(sl_r_alive)

# calculate the mean at each time step for each model
sl_r_alive = np.stack(sl_r_stack, axis=0)
sl_r_means = np.mean(sl_r_alive, axis=1)
sl_r_ses = np.std(sl_r_means, axis=0) / np.sqrt(sl_r_means.shape[0])


# same for game of life
print("Loading Game of Life robustness rules...")
gol_r_stack = []
checked = []
for gol_r in tqdm(range(gol_robust)):

    with np.load(f"data/time_series/gol_robust_{gol_r}_time_series.npz") as gol_r_data:
        gol_r_alive = np.zeros((n_inits, m_steps))
        for i in range(n_inits):
            trial_ts = gol_r_data[str(i)]
            gol_r_alive[i] = np.mean(trial_ts, axis=(1, 2))
        gol_r_stack.append(gol_r_alive)

gol_r_alive = np.stack(gol_r_stack, axis=0)
gol_r_means = np.mean(gol_r_alive, axis=1)
gol_r_ses = np.std(gol_r_means, axis=0) / np.sqrt(gol_r_means.shape[0])

# gol line with error
fig, ax = plt.subplots(ncols=2, sharey=True, figsize=(8, 4))
ax[0].plot(gol_means, label='Game of Life', c='C0')
ax[0].fill_between(range(m_steps), gol_means, gol_boots_high, 
                 linestyle='--', color='C0', alpha=0.4, label="GoL 95% CI")
ax[0].fill_between(range(m_steps), gol_boots_low, gol_means, 
                 linestyle='--', color='C0', alpha=0.4)


# dormant line with error
ax[0].plot(sl_means, label='Spore Life', c='C1')
ax[0].fill_between(range(m_steps), sl_means, sl_boots_high, 
                 linestyle='--', color='C1', alpha=0.4, label="SL 95% CI")
ax[0].fill_between(range(m_steps), sl_boots_low, sl_means, 
                 linestyle='--', color='C1', alpha=0.4)

# robustness plots
ax[1].plot(np.mean(gol_r_means, axis=0), c="C0", label="Game of Life robust mean")
ax[1].plot(np.mean(gol_r_means, axis=0) + gol_r_ses, c="C0", ls='--')
ax[1].plot(np.mean(gol_r_means, axis=0) - gol_r_ses, c="C0", ls='--',
                 label=r"GoL robust mean $\pm$ S.E.")

ax[1].plot(np.mean(sl_r_means, axis=0), c="C1", label="Spore Life robust mean")
ax[1].plot(np.mean(sl_r_means, axis=0) + sl_r_ses, c="C1", ls='--')
ax[1].plot(np.mean(sl_r_means, axis=0) - sl_r_ses, c="C1", ls='--',
                 label=r"SL robust mean $\pm$ S.E.")

ax[0].legend()
ax[1].legend()
ax[0].set_xlabel('Time (steps)')
ax[0].set_ylabel(r'Fraction of Cells in ALIVE state')
ax[1].set_xlabel('Time (steps)')
ax[1].set_ylabel(r'Fraction of Cells in ALIVE state')
plt.tight_layout()
plt.savefig('plots/living_compare/living_cells_compare.png')
plt.savefig('plots/living_compare/living_cells_compare.pdf')
plt.savefig('plots/living_compare/living_cells_compare.svg')
