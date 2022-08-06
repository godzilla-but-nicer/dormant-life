import matplotlib.pyplot as plt
import numpy as np
import os
import time
start_time = time.time()

trials = 20000
max_steps = 10000
grid_size = 5
models = 500

null_transients = np.zeros((models, trials))
gol_transients = np.zeros(trials)

# time the python execution
for i in range(models):
    os.system(f"./carust -r 3 --transient {grid_size} {max_steps} {trials}")
    os.system(f"mv data/transients_{grid_size}_{max_steps}_{trials}.csv data/null_2_transients_{grid_size}_{max_steps}_{trials}.csv")
    
    with open(f"data/null_2_transients_{grid_size}_{max_steps}_{trials}.csv", "r") as fin:
        str_vals = fin.read().split(',')[:-1]
        vals = np.array([int(val) for val in str_vals])
    
    null_transients[i] = vals

# run the actual game of life
os.system(f"./carust -f 'rule_tables/spore_life.txt' --transient {grid_size} {max_steps} {trials}")
os.system(f"mv data/transients_{grid_size}_{max_steps}_{trials}.csv data/null_3_transients_{grid_size}_{max_steps}_{trials}.csv")
with open(f"data/null_3_transients_{grid_size}_{max_steps}_{trials}.csv", "r") as fin:
    str_vals = fin.read().split(',')[:-1]
    gol_vals = np.array([int(val) for val in str_vals])

null_means = np.mean(null_transients, axis=1)
gol_mean = np.mean(gol_vals)
p_value = np.mean(gol_mean < null_means)

plt.title(f"Spore Life P-value: {p_value:3f}")
plt.hist(null_means, label="Null Model", bins=30)
plt.axvline(gol_mean, color="C1")
plt.xlabel('Mean Transient')
plt.ylabel('Count')
plt.savefig("plots/null_sl.png")
print("complete in:", time.time() - start_time)