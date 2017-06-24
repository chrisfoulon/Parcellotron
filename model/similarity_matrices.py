# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp

def similarity_correlation(co_mat):
    """ Compute the correlation matrix of a given connectivity matrix
    Parameters
    ----------
    co_mat : 2D np.array
        connectivity matrix (2D matrix seed-by-target)
    Returns
    -------
    cor_mat : 2D np.array
        seed-by-seed matrix of correlation between each and each other seed's
        connectivity profile
    """
    cor_mat = np.corrcoef(co_mat)
    ind = np.where(np.isnan(cor_mat))
    cor_mat[ind]=0

    return cor_mat



def similarity_covariance(co_mat):
    """ Compute the covariance matrix of a given connectivity matrix
    Parameters
    ----------
    co_mat : 2D np.array
        connectivity matrix (2D matrix seed-by-target)
    Returns
    -------
    cov_mat : 2D np.array
        seed-by-seed matrix of covariance between each and each other seed's
        connectivity profile
    """
    cov_mat = np.cov(co_mat)

    return cov_mat


def similarity_distance(co_mat):
    """ Compute the Euclidean distance matrix of a given connectivity matrix
    Parameters
    ----------
    co_mat : 2D np.array
        connectivity matrix (2D matrix seed-by-target)
    Returns
    -------
    cor_mat : 2D np.array
        seed-by-seed matrix of Euclidean distance between each and each other
        seed's connectivity profile
    """
    dist_mat = sp.spatial.distance_matrix(co_mat,co_mat)

    return dist_mat
