"""
A simple QOL script to concatenate all the self-play games into one file.
"""

import os
import numpy as np

files = os.listdir()
files = [f for f in files if f.startswith('self_games_')]
np_arrays = []
for f in files:
    np_arrays.append(np.load(f))
np_array = np.concatenate(np_arrays)
np.save('self_games.npy', np_array)
for f in files:
    os.remove(f)