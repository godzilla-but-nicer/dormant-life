import numpy as np
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from scipy.stats import entropy, spearmanr


# bootstrap parameters
n_inits = int(sys.argv[1])
m_steps = int(sys.argv[2])
sl_robust = int(sys.argv[3])
gol_robust = int(sys.argv[4])

# rule tables that are passed to rust have a trailing column of zeros.
# I havent had time to fix it so heres a hack
def load_rule_table(path):
    return np.loadtxt(path, delimiter=",")[:, :-1]

# I have to do this several times so i put it in a function
def load_time_series(path, on_state):
    with np.load(path) as ts_data:
        ts_alive = np.zeros((n_inits, m_steps))
        for i in range(n_inits):
            trial_ts = ts_data[str(i)]
            trial_alive = trial_ts == on_state
            ts_alive[i] = np.mean(trial_alive, axis=(1, 2))
    
    return ts_alive


# load the 
sl_rules = load_rule_table("rule_tables/rust/sl_rust.txt")
sl_ts = load_time_series("data/time_series/sl_time_series.npz", 2)


m3_rules = []
m3_ts = []

print("Loading Spore Life robustness time series...")
for sl_r in tqdm(range(sl_robust)):
    m3_ts.append(load_time_series(f"data/time_series/sl_robust_{sl_r}_time_series.npz", 2))
    m3_rules.append(load_rule_table(f"rule_tables/spore_life_robustness/random_{sl_r}.txt"))

# get the average fraction of alive cells for each model
m3_means = np.mean(np.stack(m3_ts, axis=0)[:, :, -1], axis=1)


# load the 
gol_rules = load_rule_table("rule_tables/rust/gol_rust.txt")
gol_ts = load_time_series("data/time_series/gol_time_series.npz", 2)


m2_rules = []
m2_ts = []

print("Loading Spore Life robustness time series...")
for gol_r in tqdm(range(gol_robust)):
    m2_ts.append(load_time_series(f"data/time_series/gol_robust_{gol_r}_time_series.npz", 1))
    m2_rules.append(load_rule_table(f"rule_tables/game_of_life_robustness/random_{gol_r}.txt"))

# get the average fraction of alive cells for each model
m2_means = np.mean(np.stack(m2_ts, axis=0)[:, :, -1], axis=1)
print(m3_means.shape)
print(m2_means.shape)
# combine the living means into one
m_means = deepcopy(m3_means)
m_means = np.hstack((m3_means, m2_means))


# calculate the summary stats of the rule tables
def table_dead_prob(table):
    return np.mean(table == 0)

def table_entropy(table, states):
    _, counts = np.unique(table, return_counts=True)
    freqs = counts / table.size
    return entropy(freqs) / entropy([1/states]*states)

def table_roughness(table):
    r_count = 0
    for i in range(table.shape[0]):
        for j in range(table.shape[1] - 1):
            if table[i, j] != table[i, j+1]:
                r_count += 1

    return r_count / (table.shape[0] * (table.shape[1] - 1))

# calculate the rule table features
m3_off_probs = np.array([table_dead_prob(tab) for tab in m3_rules])
m2_off_probs = np.array([table_dead_prob(tab) for tab in m2_rules])
m_off_probs = np.hstack((m3_off_probs, m2_off_probs))
print(m_off_probs.shape)


sl_mean_living = np.mean(sl_ts, axis=0)[-1]
gol_mean_living = np.mean(gol_ts, axis=0)[-1]

gol_off_prob = table_dead_prob(gol_rules)
sl_off_prob = table_dead_prob(sl_rules)


# calculate the correlation of the rule table features with the living cells
off_prob_r, off_prob_p = spearmanr(m_off_probs, m_means)

fig, ax = plt.subplots()
ax.scatter(m2_off_probs, m2_means, c="C0", label="Random Two-State", marker='x')
ax.scatter(m3_off_probs, m3_means, c="C1", label="Random Three-State", marker='x')
ax.plot(gol_off_prob, gol_mean_living, "o", mfc="C0", label="Game of Life", mec='k')
ax.plot(sl_off_prob, sl_mean_living, "o", mfc="C1", label="Spore Life", mec='k')
ax.legend()
ax.set_ylabel("Fraction ALIVE at 150 steps")
ax.set_xlabel("DEAD probability")
if off_prob_p < 0.05:
    ax.text(0.56, 0.35, r"$\rho={:.3f}$*".format(off_prob_r), ha="center", va="center")
else:
    ax.text(0.56, 0.35, r"$\rho={:.3f}$".format(off_prob_r), ha="center", va="center")

plt.tight_layout()
plt.savefig(f"plots/living_table_features/correlation_dead_prob.pdf")
plt.savefig(f"plots/living_table_features/correlation_dead_prob.png")
plt.savefig(f"plots/living_table_features/correlation_dead_prob.svg")
