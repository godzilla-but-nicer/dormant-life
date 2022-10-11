import numpy as np
from casim.Totalistic2D import GameOfLife, DormantLife
from matplotlib import animation

initial_conditions = np.random.choice([0, 2], p=[0.5, 0.10], size=[50, 50])
# initial_conditions = np.zeros((100, 100))
# initial_conditions[20:80, 20:80] = 2
gol_cond = initial_conditions == 2

gol = GameOfLife(noise=0.2, seed=420)
hist = gol.simulate(gol_cond, 200)
