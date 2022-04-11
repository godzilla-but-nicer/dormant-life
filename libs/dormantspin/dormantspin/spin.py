from cgi import test
from matplotlib.pyplot import grid
import numpy as np
from collections.abc import Callable
from typing import Iterable, Union


class Spin:
    def __init__(self, hamiltonian: Callable[[np.array, dict], float],
                 pars: dict,
                 grid_size: int,
                 p: Iterable[float] = [0.5, 0.5],
                 states: Iterable[int] = [-1, 1],
                 random_seed: Union[bool,int] = False):
        """ Generic class for spin models with arbtrary states and """
        # simple params to initiate
        self.H = hamiltonian
        self.pars = pars
        self.states = states
        
        # set up the rng
        if type(random_seed) == int:
            self.rng = np.random.default_rng(random_seed)
        else:
            self.rng = np.random.default_rng()

        # need to initialize the last energy attriubute
        self.grid = self.rng.choice(self.states,
                                    size=(grid_size, grid_size),
                                    p=p)
        self.last_energy = self.H(self.grid, self.pars)

        # we'll also initialize beta here even though this should be part of
        # metropolis()
        self.beta = 1


    def _step(self, i: int = None, j: int = None):
        """ Single step of the metropolis algorithm
            parameters i and j are for testing only """

        # randomly select a grid site
        if i is None and j is None:
            i, j = self.rng.choice(self.grid.shape[0], size=2)

        # assemble list of states that could be switched to
        valid_states = [st for st in self.states if st != self.grid[i, j]]
        
        # calculate the energy of the flip
        test_grid = self.grid.copy()
        test_grid[i, j] = self.rng.choice(valid_states)
        self.energy = self.H(test_grid, self.pars)
        delta_E = self.energy - self.last_energy

        # if a reduction in energy we will accept the change
        if delta_E <= 0:
            self.grid = test_grid
            self.last_energy = self.energy
        # otherwise its stochastic
        elif self.rng.uniform() < np.exp(- self.beta * delta_E):
            self.grid = test_grid
            self.last_energy = self.energy
        
        # also probably just for testing
        return delta_E


    def metropolis(self, t_steps: int, beta: float) -> np.array:
        """ Run the Metropolis algorithm """
        # beta parameter for simulation
        self.beta = beta

        # this will be the output array
        time_series = np.zeros((t_steps, 
                                self.grid.shape[0], 
                                self.grid.shape[1]))
        
        # run t steps
        for t in range(t_steps):
            time_series[t, :, :] = self.grid.copy()
            _ = self._step()
        
        return time_series

    def set_grid(self, grid):
        self.grid = grid
        self.last_energy = self.H(self.grid, self.pars)
        # again basically for testing
        return self.last_energy

