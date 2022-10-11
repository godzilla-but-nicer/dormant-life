import matplotlib.pyplot as plt
import numpy as np
import os


# the number of rulesets in each category
NUM_RANDOM = 53
NUM_CONNECTED = 16

# parameters for the simulation
grid = 100
steps = 100
runs = 100

r_living_fraction = np.zeros((NUM_RANDOM, runs, steps))
c_living_fraction = np.zeros((NUM_CONNECTED, runs, steps))


# random changes
for rule_i in range(NUM_RANDOM):
    os.system(f"./carust -f rule_tables/spore_life_robustness/random_{rule_i}.txt {grid} {steps} {runs} --save-all")
    for run_j in range(runs):
        run_ts = np.load(f"data/ts/time_series_{run_j}.npy")
        r_living_fraction[rule_i, run_j] = (run_ts == 2).mean(axis=(1,2))

    print(f"Rule {rule_i} complete!")

# connected changes
for rule_i in range(NUM_CONNECTED):
    os.system(f"./carust -f rule_tables/spore_life_robustness/connected_{rule_i}.txt {grid} {steps} {runs} --save-all")
    for run_j in range(runs):
        run_ts = np.load(f"data/ts/time_series_{run_j}.npy")
        c_living_fraction[rule_i, run_j] = (run_ts == 2).mean(axis=(1,2))

    print(f"Connected rule {rule_i} complete!")

r_rule_living = r_living_fraction.mean(axis=1)
np.save("data/robustness_random_living.npy", r_rule_living)

c_rule_living = c_living_fraction.mean(axis=1)
np.save("data/robustness_connected_living.npy", c_rule_living)

print(r_rule_living.shape)
    
plt.plot(c_rule_living.T, color="C1", label="Connected", lw=2)
plt.plot(r_rule_living.T, color="C0", label="Random", lw=2)
plt.legend()
plt.xlabel("Time (steps)")
plt.ylabel("Living Fraction")
plt.savefig("plots/robustness_living_out.png")
