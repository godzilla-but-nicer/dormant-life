#%%
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import casim.Totalistic2D as cas
import sys
from spore_life_rule_table import sl_rules

#sl = cas.DormantLife()
#stg = sl.get_state_transition_graph((3,3))
model = sys.argv[1]

stg = nx.read_graphml(f"data/stgs/{model}.graphml")
# number of attractors
attractors = list(nx.simple_cycles(stg))

# size of largest basin
basins = nx.weakly_connected_components(stg)
ccs = sorted(basins, key=len, reverse=True)
cc_size = [len(cc) for cc in ccs]
print(cc_size[0])

# largest basin is extinction
extinction = '0' in ccs[0]
print(extinction)

# standard deviation in degree??
in_deg = np.array(list(stg.in_degree()), dtype=float)[:, 1]
print(in_deg)
print(in_deg.std())

# transient lengths
t_len = []
ext_t_len = []

cycles = nx.simple_cycles(stg)
# need to do this for each attractor
for cycle in cycles:
    # set to ignore
    cyc_nodes = set(cycle)
    for node in cycle:
        # initialize the queue and our distance counter
        queue = [node]
        level = {node: 0}
        # keep going til the queue is done
        while queue:
            # remove the first element, thats what we check next
            base = queue.pop(0)
            for pred in [p for p in stg.predecessors(base) if p not in cyc_nodes]:
                queue.append(pred)
                # we know that this must be one step farther out than the checked node
                level[pred] = level[base] + 1
        t_len.extend([l for l in level.values()])
        if '0' in cyc_nodes:
            ext_t_len.extend(l for l in level.values())
        
with open(f"data/stgs/stg_{model}_out.txt", "w") as fout:
    fout.write(f"total num states: {len(stg.nodes)}\n")
    fout.write(f"num attractors: {len(attractors)}\n")
    fout.write(f"largest connected component: {cc_size[0]}\n")
    fout.write(f"normalized connected component: {cc_size[0] / len(stg.nodes)}\n")
    fout.write(f"extinction is largest: {extinction}\n")
    fout.write(f"in degree s.d.: {in_deg.std()}\n")
    fout.write(f"average transient length: {np.mean(t_len)}")
    fout.write(f"average transient to ext.: {np.mean(ext_t_len)}")

# %%
