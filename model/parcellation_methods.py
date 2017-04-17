# -*- coding: utf-8 -*-

from sklearn.cluster import KMeans

def parcellate_KMeans(sim_mat, nb_clu):
    """ Parellate a 2D similarity matrix with the KMeans algorithm
    Parameters
    ----------
    sim_mat : 2D np.array
        square similarity_matrix, e.g. correlation matrix
    nb_clu = int
        desired number of clusters
    Returns
    -------
    labels : np.array
        Nseed labels (integers) which can be used to assign to each seed ROI the
        value associated to a certain cluster
    """

    labels = KMeans(n_clusters=nb_clu).fit_predict(sim_mat)

    return labels
