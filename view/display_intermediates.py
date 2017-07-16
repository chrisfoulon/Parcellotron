import numpy as np

import matplotlib.pyplot as plt

# Just to keep the lines in memory, but it does not work.
def file_visualization(path):
    mat = np.load(path)
    plt.imshow(mat, aspect='auto', interpolation='none')
    plt.show()

def mat_visualization(mat):
    plt.imshow(mat, aspect='auto', interpolation='none')
    plt.show()

path="/data/BCBLab/s01pomat3_LH/Tracto_mat"
