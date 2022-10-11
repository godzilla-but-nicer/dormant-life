import casim.Totalistic2D as ct
import casim.utils as cu
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# calculate the number of states and set up the transition table
grid = [4, 3]
s = [2, 3]
num_states = [2**16, 3**9]
graph_name = ["gol", "sl"]
rule_table = ["rule_tables/game_of_life.txt", "rule_tables/spore_life.txt"]

for i, model in enumerate(range(len(graph_name))):

    # initialize the ca with the rule table
    table = np.loadtxt(rule_table[i], delimiter=',')
    ca = ct.Totalistic2D(s[i], table)

    # this will be the edge list from which we make our graph
    edges = []

    for dec_state in tqdm(range(num_states[i])):
        # set the state and get the next step
        state = cu.to_base(dec_state, s[i], grid[i]**2).reshape((grid[i], grid[i]))
        
        next = ca.step(state)

        # now we need to back encode the grid into a number
        dec_next = cu.to_decimal(next.flatten(), grid[i]**2, s[i])

        edges.append((dec_state, dec_next))

    
    print(len(edges))
    print(np.max(edges))
    STG = nx.from_edgelist(edges, create_using=nx.DiGraph)

    nx.write_graphml(STG, f"data/stgs/{graph_name[i]}.graphml")
