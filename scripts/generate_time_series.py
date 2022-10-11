import numpy as np
import os
import sys
from glob import glob
from tqdm import tqdm

# small grids for transient comparisons
runs = sys.argv[1]  # 1000
max_steps = sys.argv[2]  # 10000
t_grid = sys.argv[3]  # 6

# specific rules
print("Running transients for Spore Life...")
os.system(f"./carust -f rule_tables/rust/sl_rust.txt --transient {t_grid} {max_steps} {runs}")
os.system(f"mv data/transients.npy data/transients/sl_transients.npy")
print("Running transients for Game of Life...")
os.system(f"./carust -f rule_tables/rust/gol_rust.txt --transient {t_grid} {max_steps} {runs}")
os.system(f"mv data/transients.npy data/transients/gol_transients.npy")

# random models
random_models = sys.argv[4]  # 100
print("Running transients for random models...")
for i in tqdm(range(random_models)):
    os.system(f"./carust -r 3 --transient --save-rules {t_grid} {max_steps} {runs}")
    os.system(f"mv data/transients.npy data/transients/r{i}_3_transients.npy")
    os.system(f"mv data/rule_table.npy data/random_rules/r{i}_3_rules.npy")
    os.system(f"./carust -r 2 --transient --save-rules {t_grid} {max_steps} {runs}")
    os.system(f"mv data/transients.npy data/transients/r{i}_2_transients.npy")
    os.system(f"mv data/rule_table.npy data/random_rules/r{i}_2_rules.npy")

# large grids for living cell comparisons
l_steps = sys.argv[5]  # 150
l_grid = sys.argv[6]  # 20
sl_r = int(sys.argv[7])  # 23
gol_r = int(sys.argv[8]) # 15
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
