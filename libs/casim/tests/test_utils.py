import numpy as np
from casim.utils import to_binary, to_decimal, to_base


def test_to_binary():
    assert np.array_equal(to_binary(54), [0, 0, 1, 1, 0, 1, 1, 0])


def test_to_decimal():
    assert to_decimal(np.array([0, 0, 1, 1, 0, 1, 1, 0]), 8) == 54

def test_to_base():
    assert np.array_equal([2, 2, 2, 1, 2, 0, 2], to_base(2153, 3, 7))
