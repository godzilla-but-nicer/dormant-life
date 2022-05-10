from casim.Totalistic2D import DormantLife, GameOfLife
import matplotlib.pyplot as plt
import numpy as np

# fixed parameters
rng_seed = 666
rng = np.random.default_rng(rng_seed)
grid_size = 20
time_steps = 200
trials = 40

# noise parameters
num_etas = 100
etas = np.linspace(0., 1., num_etas).astype(float)

avg_gol_living = np.zeros(etas.shape[0])
sd_gol_living = np.zeros(etas.shape[0])
avg_dl_living = np.zeros(etas.shape[0])
sd_dl_living = np.zeros(etas.shape[0])

# we want to forbid i -> i noise transitions in both models
gol_t = np.array([[0, 1],
                  [1, 0]])
dl_t = np.array([[0.0, 0.5, 0.5],
                 [0.0, 0.0, 1.0],
                 [0.5, 0.5, 0.0]])


for ni, eta in enumerate(etas):
    trial_gol_living = np.zeros(trials)
    trial_dl_living = np.zeros(trials)
    for ti in range(trials):
        dl = DormantLife(noise=eta, transitions=dl_t, seed=rng_seed)

        initl = rng.choice([0, 2], size=(grid_size, grid_size))
        dl_hist = dl.simulate(initl, time_steps)

        trial_dl_living[ti] = np.sum(dl_hist[-1] == 2)
    
    # summary stats for the noise level
    avg_dl_living[ni] = np.mean(trial_dl_living)
    sd_dl_living[ni] = np.std(trial_dl_living)

fig, ax = plt.subplots()
ax.fill_between(etas, avg_dl_living - sd_dl_living, 
                avg_dl_living + sd_dl_living,
                alpha=0.4, c='C1')
ax.plot(etas, avg_dl_living, label='DormantLife', c='C1')

ax.legend()
ax.set_xlabel(r'Noise, $\eta$')
ax.set_ylabel(r'Number of Living Cells $\pm$ S.D.')
plt.savefig('plots/noise_living_dormancy.png')
plt.savefig('plots/noise_living_dormancy.pdf')






