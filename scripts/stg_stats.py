import networkx as nx
import numpy as np
import casim.Totalistic2D as cas
from spore_life_rule_table import sl_rules

sl = cas.Totalistic2D(3, sl_rules)
stg = sl.get_state_transition_graph((3,3))

# transient length
attractors = list(nx.simple_cycles(stg))
for 
print(attractors)
# number of connected components
ccs = len(attractors)

# size of connected component

# avg in degree??