import numpy as np
import numpy.typing as npt
import networkx as nx
from typing import Iterable, Tuple, Union
from scipy import signal
from .utils import to_base, to_decimal


class Totalistic2D:
    def __init__(self, n_states: int, thresholds: Iterable[Iterable[Iterable]],
                 noise: Union[float, Iterable[float]] = 0.0,
                 transitions: Union[Iterable[Iterable[float]], bool] = False,
                 seed: int = None):
        """
        This class contains functions for 2D CA models that depend on the
        number of neighbors but not their specific arrangement
        """

        # set the possible states and thresholds for the states
        self.thresholds = np.zeros(n_states)
        self.states = np.arange(n_states)
        self.filter = np.array([[1, 1, 1],
                                [1, 0, 1],
                                [1, 1, 1]])

        # thresholds is the set of integers for which state i
        # transitions to state j
        self.thresholds = thresholds

        self._init_stochasticity(n_states, noise, transitions, seed)

    def simulate(self, init_grid, steps: int):
        """
        this method simulates the thing
        """
        if type(init_grid) == int:
            self.grid = self.rng.choice([0, 1], size=[init_grid, init_grid])
        else:
            self.grid = init_grid

        self.history = np.zeros(
            (steps+1, self.grid.shape[0], self.grid.shape[0]))

        for st in range(steps):
            self.history[st] = self.grid
            self.grid = self.step(self.grid)

        self.history[-1] = self.grid

        return self.history

    def simulate_transients(self, init_grid, max_steps: int):
        """
        this method simulates the thing until an attractor is found
        """
        all_history = self.simulate(init_grid, max_steps)

        # find the attractor by looking backwards
        found_attractor = False
        for z, past_state in enumerate(all_history[::-1]):
            if np.array_equal(past_state, all_history[-1]) and z > 0:
                idx = all_history.shape[0] - z
                found_attractor = True
                break

        # the way this in indexed should get JUST one of each state in the
        # attractor. It at least works with period 1 attractors. Might not
        # on longer attractors??
        # iterate from the beginning to see where attractor first appears
        if not found_attractor:
            return all_history, np.nan
        else:
            attractor = all_history[idx:]
            for i, past_state in enumerate(all_history):
                for attr_state in attractor:
                    if np.array_equal(past_state, attr_state):
                        return (all_history[:i], i-1)

    def step(self, grid: npt.ArrayLike) -> npt.ArrayLike:
        """
        generic step function
        """
        neighbors = signal.convolve2d(grid, self.filter,
                                      mode='same', boundary='wrap')
        new_grid = grid.copy()

        noisy, noise_grid = self._resolve_noise(new_grid)
        grid[noisy] = noise_grid[noisy]

        for i in range(self.states.shape[0]):
            for j in range(self.states.shape[0]):
                for k in self.thresholds[i][j]:
                    new_grid[(grid == i) & (neighbors == k)] = j

        return new_grid

    def get_state_transition_graph(self, size: tuple) -> nx.DiGraph:
        edges = []
        sites = size[0] * size[1]
        num_configs = self.states.shape[0] ** sites

        for dec_state in range(num_configs):
            # set the state and get the next step
            state = to_base(dec_state, 3, 9).reshape((3, 3))
            next = self.step(state)

            # now we need to back encode the grid into
            dec_next = to_decimal(next.flatten(), 9, 3)

            edges.append((dec_state, dec_next))
        
        D = nx.from_edgelist(edges, create_using=nx.DiGraph)
        return D
        


    def _resolve_noise(self, grid: npt.ArrayLike) -> Tuple[npt.ArrayLike,
                                                           npt.ArrayLike]:
        """
        this function contains all of the code to update the grid for the sites
        that are deemed noisy
        """
        filter = np.zeros(grid.shape).astype(bool)
        new_states = np.zeros(grid.shape)
        for st, eta in enumerate(self.noise):
            st_filter = (grid == st) & \
                        (self.rng.uniform(size=grid.shape) < eta)
            # we want to return a filter so we don't undo the noise changes
            filter = filter | st_filter

            # we also need to do the changes per state
            new_states[st_filter] = self.rng.choice(
                self.states,
                size=grid.shape,
                p=self.transitions[st])[st_filter]

        return (filter, new_states)

    def _init_stochasticity(self, n_states, noise, transitions, seed):
        # noise thresholds
        if type(noise) == float or type(noise) == np.float64:
            self.noise = np.repeat(noise, self.states.shape[0])
        elif type(noise) == Iterable:
            self.noise = np.array(noise)
        else:
            raise(TypeError, "noise is a bad type")

        # if not given the transition matrix is uniform
        if type(transitions) == bool:
            self.transitions = (np.repeat(1/n_states, n_states**2)
                                .reshape((n_states, n_states)))
        else:
            self.transitions = transitions

        # rng object
        if seed:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()


class GameOfLife(Totalistic2D):
    def __init__(self,
                 noise: Union[float, Iterable[float]] = 0.0,
                 transitions: Union[Iterable[Iterable[float]], bool] = False,
                 seed: int = None):
        """
        this class simulates conway's game of life
        """
        # set params
        self.states = np.array([0, 1])
        self.survive = {'low': 2, 'high': 3}
        self.reproduce = 3

        # set conv filter
        self.filter = np.array([[1, 1, 1],
                                [1, 0, 1],
                                [1, 1, 1]])

        self._init_stochasticity(self.states.shape[0], noise, transitions,
                                 seed)

    def step(self, grid):
        """
        this does a single step. we're going to do convolution!!
        """
        neighbors = signal.convolve2d(grid, self.filter,
                                      mode='same', boundary='wrap')
        new_grid = grid.copy()

        # _ is a boolean mask of which cells flipped
        noisy, noise_grid = self._resolve_noise(new_grid)
        grid[noisy] = noise_grid[noisy]

        # survival
        new_grid[(grid == 1) &
                 ((neighbors < self.survive['low']) |
                  (neighbors > self.survive['high']))] = 0

        # reproduction
        new_grid[(grid == 0) & (neighbors == self.reproduce)] = 1

        return new_grid


class DormantLife(Totalistic2D):
    def __init__(self,
                 noise: Union[float, Iterable[float]] = 0.0,
                 transitions: Union[npt.ArrayLike, bool] = False,
                 seed: int = None):
        """
        this class implements the three state game of life described in
        Javid 2007
        """
        # set params
        self.states = np.array([0, 1, 2])
        self.survive = {'low': 2, 'high': 3}
        self.reproduce = 3
        self.die = 4

        # three states {alive: 2, dormant: 1, dead: 0}

        # set conv filter
        self.filter = np.array([[1, 1, 1],
                                [1, 0, 1],
                                [1, 1, 1]])

        self._init_stochasticity(self.states.shape[0], noise,
                                 transitions, seed)

    def step(self, grid):
        """
        single step of the dormancy CA using convolution
        """
        # copy the grid to make a new grid
        new_grid = grid.copy()

        # this mask ensures thats the checks on dormant cells occur correctly
        updated = np.zeros(new_grid.shape, dtype=bool)

        # need 2 arrays to convolve
        dormant = grid == 1
        alive = grid == 2

        # convolve!
        d_neighbors = signal.convolve2d(
            dormant, self.filter, mode='same', boundary='wrap')
        a_neighbors = signal.convolve2d(
            alive, self.filter, mode='same', boundary='wrap')

        # i spent a lot of time ensuring we dont process the noisy grid
        # but actually I think its fine?
        noisy, noise_grid = self._resolve_noise(new_grid)
        grid[noisy] = noise_grid[noisy]

        # sporulation
        new_grid[(grid == 2) &
                 ((a_neighbors < self.survive['low']) |
                  (a_neighbors > self.survive['high']))] = 1

        # reproduction
        new_grid[(grid == 0) & (a_neighbors == self.reproduce)] = 2

        # dormant dying
        new_grid[(grid == 1) &
                 (d_neighbors + a_neighbors > self.die)] = 0
        updated[(grid == 1) &
                (d_neighbors + a_neighbors > self.die)] = True

        # dormant awakening
        new_grid[(np.equal(updated, False)) &  # flake8 friendly
                 (grid == 1) &
                 ((d_neighbors >= self.survive['low']) &
                  (d_neighbors <= self.survive['high']))] = 2

        return new_grid
