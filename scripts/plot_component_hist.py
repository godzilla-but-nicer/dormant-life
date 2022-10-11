#%%
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

point_colors = np.array(["C0", "C1"])
names = ["gol", "sl"]
for i in range(2):
    print(i)
    STG = nx.read_graphml(f"../data/stgs/{names[i]}.graphml")

    # make plot for distribution of component sizes
    # first we need to calculate them
    ccs = nx.weakly_connected_components(STG)
    sizes = []
    for cc in ccs:
        sizes.append(len(cc))
    sizes = np.array(sizes)
    vals, counts = np.unique(sizes, return_counts=True)

    # set up histogram parameters
    num_bins = 10
    bins = np.logspace(0, 5, num_bins)
    Y, X = np.histogram(sizes, bins)
    pdf = Y / np.sum(Y)

    fig, ax = plt.subplots()
    ax.loglog(vals, counts, 'o', 
              markerfacecolor=point_colors[i], markeredgecolor=point_colors[i])
    ax.set_ylabel("Number of components")
    ax.set_xlabel("Size of component")
    ax.set_xlim()
    ax.set_ylim()
    plt.savefig(f"../plots/stg_comp_distr/{names[i]}.pdf")
    plt.savefig(f"../plots/stg_comp_distr/{names[i]}.png")


# %%
