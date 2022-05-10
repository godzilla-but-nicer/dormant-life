import numpy as np
from typing import Iterable, Union
from .spin import Spin


class Ising(Spin):
    def __init__(self, pars: dict, 
                 grid_size: int,
                 p: Iterable[float] = [0.5, 0.5],
                 random_seed: Union[bool,int] = False):
        """ 
        This is a class that lets us quickly set up an ising model without
        having to set up our own hamiltonian. Useful as a comparison """

        # set up the hamiltonian
        def ising_hamiltonian(xy: np.array, k: dict) -> float:
            interactions = 0.0
            external_field = 0.0
            # i need this with short name for indexing
            gs = xy.shape[0]
            for i in range(xy.shape[0]):
                for j in range(xy.shape[1]):
                    # calculated once per site
                    external_field += xy[i, j]

                    # calculated 4x per site
                    interactions += xy[i, j] * xy[(i-1) % gs, j]
                    interactions += xy[i, j] * xy[(i+1) % gs, j]
                    interactions += xy[i, j] * xy[i, (j-1) % gs]
                    interactions += xy[i, j] * xy[i, (j+1) % gs]
            
            return -k["J"]*interactions - k["h"]*external_field

        # easy
        self.H = ising_hamiltonian
        self.states = [-1, 1]
        self.pars = pars

        # set up the rng
        if type(random_seed) == int:
            self.rng = np.random.default_rng(random_seed)
        else:
            self.rng = np.random.default_rng()

        # need to initialize the last energy attribute
        self.grid = self.rng.choice(self.states,
                                    size=(grid_size, grid_size),
                                    p=p)
        self.last_energy = self.H(self.grid, self.pars)

        # we'll also initialize beta here even though this should be part of
        # metropolis()
        self.beta = 1

class IsingLife(Spin):
    def __init__(pars: dict,
                 grid_size: int,
                 p: Iterable[float]=[0.5, 0.5],
                 random_seed: Union[bool, int] = None):
        pass