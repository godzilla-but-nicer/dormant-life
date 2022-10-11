#%%
import numpy as np
import matplotlib.pyplot as plt
import os

real_living_fraction = np.zeros(100)

rr = np.load("../data/robustness_random_living.npy")
rc = np.load("../data/robustness_connected_living.npy")
os.system(f"./carust -f rule_tables/spore_life.txt 100 100 100 --save-all")
for run_j in range(100):
    run_ts = np.load(f"../data/ts/time_series_{run_j}.npy")
    real_living_fraction[run_j] = (run_ts == 2).mean(axis=(1,2))



print(rr.shape)
print(rc.shape)
# %%
for i in range(rr.shape[0]):
    if i == 0:
        plt.plot(rr[i, :], label="Random Changes", c="C1", alpha=0.6)
    else:
        plt.plot(rr[i, :], c="C1", alpha=0.6)

plt.legend()
plt.show()
# %%
