import casim.Totalistic2D
import numpy as np
import pandas as pd
import pickle
import time
from spore_life_rule_table import sl_rules

# params
start_time = time.time()
rng = np.random.default_rng(666)
grid_size = 5
rulesets = 11
initial_conditions = 10
max_steps = 10000

# output files and formatting
tables_path = "data/null_models.pkl"
null_transients_path = "data/null_transients.csv"
model_transients_path = "data/model_transients.csv"

# generate the initial conditions that all models will use
init_grids = rng.choice(range(2), size=(initial_conditions, grid_size, grid_size))
null_tables = []

for rs in range(rulesets):
    # generate rule table
    rules = [[], [], []]
    for st in range(3):
        # from each tate we transition to other states based on num neighbors
        i = rng.choice(range(3), size=8)
        for j in range(3):
            rules[st].append(np.where(i == j))
    # big list of all rule tables for saving
    null_tables.append(rules)
    
    # initialize the simulator and run 
    ca = casim.Totalistic2D.Totalistic2D(3, thresholds=rules)
    null_transients = np.zeros((rulesets, initial_conditions))
    for ic in range(initial_conditions):
        _, t_len = ca.simulate_transients(init_grids[ic], max_steps)
        null_transients[rs, ic] = t_len+1

# write out the null model data
with open(tables_path, "wb") as fpkl:
    pickle.dump(null_tables, fpkl)

null_transients_df = pd.DataFrame(null_transients, columns=[f"ic{i}" for i in range(initial_conditions)])
with open(null_transients_path, "w") as fnt:
    null_transients_df.to_csv(fnt)

# we also need to simulate our rules of interest from the same initial conditions
gol = casim.Totalistic2D.GameOfLife()
dl = casim.Totalistic2D.DormantLife()
sl = casim.Totalistic2D.Totalistic2D(3, thresholds=sl_rules)

gol_transients = np.zeros(initial_conditions)
dl_transients = np.zeros(initial_conditions)
sl_transients = np.zeros(initial_conditions)

for ic in range(initial_conditions):
    _, t_gol = gol.simulate_transients((init_grids[ic] == 2).astype(int), max_steps)
    gol_transients[ic] = t_gol+1
    _, t_dl = dl.simulate_transients(init_grids[ic], max_steps)
    dl_transients[ic] = t_dl+1
    _, t_sl = sl.simulate_transients(init_grids[ic], max_steps)
    sl_transients[ic] = t_sl+1

model_transients_df = pd.DataFrame([gol_transients, dl_transients, sl_transients],
                                   columns=[f"ic{i}" for i in range(initial_conditions)])
model_transients_df["model"] = ["gol", "dl", "sl"]
with open(model_transients_path, "w") as fout:
    model_transients_df.to_csv(fout)
