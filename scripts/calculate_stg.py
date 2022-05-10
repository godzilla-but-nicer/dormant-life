import casim.Totalistic2D as ct
import casim.utils as cu
import networkx as nx
import numpy as np

# calculate the number of states and set up the transition table
num_states = 3**9
graph_name = "GoL"
table  = [[[0, 1, 2, 4, 5, 6, 7, 8], [], [3]],
          [[], [], []],
          [[0, 1, 4, 5, 6, 7, 8], [], [2, 3]]]

# table  = [[[0, 1, 2, 4, 5, 6, 7, 8], [], [3]],
#           [[8], [0, 1, 4, 5, 6, 7], [2, 3]],
#           [[0, 4, 5, 6, 7, 8], [1], [2, 3]]]



# initialize the ca with the rule table
ca = ct.Totalistic2D(3, table)

# this will be the edge list from which we make our graph
edges = []

for dec_state in range(num_states):
    # set the state and get the next step
    state = cu.to_base(dec_state, 3, 9).reshape((3, 3))
    next = ca.step(state)

    # now we need to back encode the grid into
    dec_next = cu.to_decimal(next.flatten(), 9, 3)

    edges.append((dec_state, dec_next))

STG = nx.from_edgelist(edges, create_using=nx.DiGraph)
nx.write_graphml(STG, f"data/stgs/{graph_name}.graphml")