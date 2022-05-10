import numpy as np


# converts an integer to a boolean representation as a list
def to_binary(n, digits=8):
    binary_digits = []
    for _ in range(digits):
        binary_digits.append(int(n % 2))
        n = int(n / 2)
    return np.array(binary_digits[::-1])


# generalization of to_binary
def to_base(n: int, base: int, digits: int = 8) -> np.ndarray:
    base_digits = []
    for _ in range(digits):
        base_digits.append(int(n % base))
        n = int(n / base)
    return np.array(base_digits[::-1])


# convert binary list to integer
def to_decimal(b, digits, base: int = 2) -> int:
    expos = np.arange(digits, 0, -1) - 1
    enc = base**expos
    return np.array(b).T.dot(enc)
