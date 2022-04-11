from dormantspin import Spin
import numpy as np

tester_grid = np.array([[-1,  1,  1],
                        [-1,  1, -1],
                        [-1, -1, -1]])

def th(x, k):
    return np.sum(x)

def test_init():
    sp = Spin(th, pars = {}, grid_size=3, p = [0.5, 0.5], 
              states = [-1, 1], random_seed=1)
    xx = sp.H(np.array([1, 1, -1]), {})
    assert(xx == 1)

def test_set_grid():
    sp = Spin(th, pars = {}, grid_size=3, p = [0.5, 0.5], 
              states = [-1, 1], random_seed=1)
    le = sp.set_grid(tester_grid)
    assert(le == -3)

def test_step():
    sp = Spin(th, pars = {}, grid_size=3, p = [0.5, 0.5], 
              states = [-1, 1], random_seed=1)
    _ = sp.set_grid(tester_grid)
    de = sp._step(1, 1)
    assert(de == -2)

def test_metropolis():
    sp = Spin(th, pars = {}, grid_size=3, p = [0.5, 0.5], 
              states = [-1, 1], random_seed=1)
    sp.metropolis(100, 1000)
    energy = sp.H(sp.grid, {})
    assert(energy == -9)
    