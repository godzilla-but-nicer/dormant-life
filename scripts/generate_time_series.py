import numpy as np
import os
import sys
from glob import glob
from tqdm import tqdm

# unpack all of the parameters used from workflow/config.yaml
# medium grids for transient comparisons
threads = int(sys.argv[1])    # transients["threads"]
t_grid = int(sys.argv[2])     # transients["t_grid"]
max_steps = int(sys.argv[3])  # transients["max_steps"]
runs = int(sys.argv[4])       # transients["runs"]

# small grids for null model transient comparisons
random_models = int(sys.argv[5])   # null_models["random_models"]
null_t_grid = int(sys.argv[6])     # null_models["t_grid"]
null_max_steps = int(sys.argv[7])  # null_models["max_steps"]
null_runs = int(sys.argv[8])       # null_models["runs"]

# large grids for living cell comparisons
l_steps = int(sys.argv[9])  # living_cells["steps"]
l_grid = int(sys.argv[10])  # living_cells["l_grid"]
sl_r = int(sys.argv[11])    # robustness["sl_len"]
gol_r = int(sys.argv[12])   # robustness["gol_len"]

# specific rules
print("Running transients for Spore Life...")
os.system(f"./carust -f rule_tables/spore_life.txt -t {threads} --transient {t_grid} {max_steps} {runs}")
os.system(f"mv data/transients.npy data/transients/sl_transients.npy")
print("Running transients for Game of Life...")
os.system(f"./carust -f rule_tables/game_of_life.txt -t {threads} --transient {t_grid} {max_steps} {runs}")
os.system(f"mv data/transients.npy data/transients/gol_transients.npy")

# random models
print("Running transients for random models...")
for i in tqdm(range(random_models)):
    os.system(f"./carust -r 3 -t {threads} --transient --save-rules {null_t_grid} {max_steps} {runs}")
    os.system(f"mv data/transients.npy data/transients/r{i}_3_transients.npy")
    os.system(f"mv data/rule_table.npy data/random_rules/r{i}_3_rules.npy")
    os.system(f"./carust -r 2 -t {threads} --transient --save-rules {null_t_grid} {max_steps} {runs}")
    os.system(f"mv data/transients.npy data/transients/r{i}_2_transients.npy")
    os.system(f"mv data/rule_table.npy data/random_rules/r{i}_2_rules.npy")

# number of living cells time series
print("Running Spore Life time series...")
os.system(f"./carust -f rule_tables/rust/sl_rust.txt --save-all {l_grid} {l_steps} {runs}")
os.system(f"mv data/time_series.npz data/time_series/sl_time_series.npz")
print("Running Game of Life Time series...")
os.system(f"./carust -f rule_tables/rust/gol_rust.txt --save-all {l_grid} {l_steps} {runs}")
os.system(f"mv data/time_series.npz data/time_series/gol_time_series.npz")

# robustness models
print("Running spore life robustness time series...")
for i in tqdm(range(sl_r)):
    os.system(f"./carust -f rule_tables/spore_life_robustness/random_{i}.txt --save-all {l_grid} {l_steps} {runs}")
    os.system(f"mv data/time_series.npz data/time_series/sl_robust_{i}_time_series.npz")
print("Running game of liferobustness time series")
for i in tqdm(range(gol_r)):
    os.system(f"./carust -f rule_tables/game_of_life_robustness/random_{i}.txt --save-all {l_grid} {l_steps} {runs}")
    os.system(f"mv data/time_series.npz data/time_series/gol_robust_{i}_time_series.npz")
