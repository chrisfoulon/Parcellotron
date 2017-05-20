import numpy as np

import matplotlib.pyplot as plt

# Just to keep the lines in memory, but it does not work.
def temp_visualization(path):
    mat = np.load(path)
    plt.imshow(mat, aspect='auto', interpolation='none');
    plt.show()
