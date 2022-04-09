import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
from casim.Totalistic2D import GameOfLife

# steps to simulate
t_trials = 10
m_steps = 1000
grid_size = 100
noise = 0.00

# analytical solution
def p_analytical(k, l, n):
    numerator = 2 * k - ((k + l - 1)*(l - k)**n)
    denominator = 2 * (k - l + 1)
    return numerator / denominator


on_to_on = (comb(8, 2) + comb(8, 3)) / 2**8
off_to_on = (comb(8, 3)) / 2**8
print(p_analytical(off_to_on, on_to_on, 10))

# set up the simulation
gol = GameOfLife(420)
series = np.zeros((m_steps, t_trials))

# run that shit
for ti in range(t_trials):
    gol_grid = np.random.choice(2, size=[grid_size, grid_size])
    for si in range(m_steps):
        flip = np.random.uniform(size=[grid_size, grid_size]) < noise
        # print(np.vstack(np.where(flip)))
        for (i, j) in np.vstack(np.where(flip)).T:
            if gol_grid[i, j] == 1:
                gol_grid[i, j] = 0
            else:
                gol_grid[i, j] = 1
        series[si, ti] = np.mean(gol_grid)
        gol_grid = gol.step(gol_grid)

avg_living = np.mean(series, axis=1)

plt.plot(range(m_steps), 
         [p_analytical(off_to_on, on_to_on, i) for i in range(m_steps)],
         c='k')
plt.scatter(range(m_steps), avg_living)
plt.show()







