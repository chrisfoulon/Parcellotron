# -*- coding: utf-8 -*-

import nibabel as nib
import numpy as np
import scipy.stats as st


def matrix_log2(matrix):
    """ Apply log in base 2 on the matrix
    Parameters
    ----------
    matrix = 2D np.array
        Typically a 2D matrix seed by target
    Returns
    -------
    matrix_log2 : 2D np.array
        log2 of connectivity_matrix + 1
    """

    matrix_log2 = np.log2(matrix + 1)

    return matrix_log2



def matrix_zscore(matrix):
    """ Apply Z-score tranformation on the matrix
    Parameters
    ----------
    matrix = 2D np.array
        Typically a 2D matrix seed by target

    Returns
    -------
    z_matrix : 2D np.array
        Zscore of connectivity_matrix, replacing each value with its Z score
        across ROIs
    """

    # Sometimes there are voxels with empty connectivity, which could be due to
    # either some NaNs in preprocessing phases or to isolation of the voxel.
    # They are typically in the order of 1/10000.
    # To deal with this, we inject random values from a gaussian (u=0,std=1),
    # so that we also don't have annoying NaNs in the similarity matrix

    # # To test the procedure, try to introduce some zero columns std
    # connectivity_matrix[:,2396:2397] = np.zeros([nROIs,1])
    # connectivity_matrix[:,85:86] = np.zeros([nROIs,1])

    nROIs = matrix.shape[0]

    ind_zerostd = np.where(np.sum(matrix, axis=0) == 0)

    if np.squeeze(ind_zerostd).any():

        numba_voxels_zerostd = np.array(ind_zerostd).shape[1]
        print("I found " + str(numba_voxels_zerostd) + " voxels with zero std.")
        print("I will replace them with normally distributed random numbers")

        matrix[:, [i for i in ind_zerostd]] =\
            np.random.randn(nROIs, 1, numba_voxels_zerostd)


    z_matrix = st.zscore(matrix, axis=0)

    return z_matrix
