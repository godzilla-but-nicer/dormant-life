import matplotlib.pyplot as plt
import numpy as np
import os
import time

trials = 100

pytimes = np.zeros(trials)
rusttimes = np.zeros(trials)

# time the python execution
for i in range(trials):
    start_time = time.time()
    os.system('python scripts/generic-game-of-life.py')
    pytimes[i] = time.time() - start_time

    start_time = time.time()
    os.system('./carust -f ../carust/rule_tables/game_of_life.txt --transient 7 10000 100')
    rusttimes[i] = time.time() - start_time

plt.figure()
plt.hist(pytimes, bins=20, alpha=0.7, label='Python')
plt.hist(rusttimes, bins=20, alpha=0.7, label='Rust')
plt.legend()
plt.savefig('plots/rust_time_compare.png')

print(pytimes.mean())
print(rusttimes.mean())