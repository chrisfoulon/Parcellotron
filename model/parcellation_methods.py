# -*- coding: utf-8 -*-

from sklearn.cluster import KMeans
from sklearn import decomposition
import numpy as np
import matrix_transformations as mt

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

def parcellate_PCA(similarity_matrix, output_folder):
    """ Parellate a 2D similarity matrix with the PCA algorithm
    Parameters
    ----------
    similarity_matrix : 2D np.array
        square similarity_matrix, e.g. correlation matrix
        (It is assumed that the original 2D_connectivity_matrix was normalized
        across ROIs)
    Returns
    -------
    labels : np.array
        Nseed labels (integers) which can be used to assign to each seed ROI the
        value associated to a certain cluster
    """
    sim_mat = similarity_matrix
    # Perform the first decomposition using svd
    # NB!!! In Python "v" is ALREADY TRANSPOSED, meaning that
    # X = U * S * VT
    u, s, vt = np.linalg.svd(sim_mat, full_matrices=True)

    # Take the transpose of vt to operate on a matrix where
    # eigenvectors/loadings are on the columns
    pc = vt.T

    # Here you can choose to consider only a subset of eigenvectors
    # for rotation
    pc = pc[:,:]

    # Rotate the "v", i.e. the principal components
    pc_rot = mt.rotate_components(pc, gamma = 0.0)

    # Go back to the original orientation. CAREFUL: in the numpy
    # implementation, "v" is already rotated! (hence the name "vt")
    vt_rot = pc_rot.T

    # Calculate the eigenvalues after rotation
    # NB: Since vt_rot is already transposed, and
    #         Sigma = U' * X * V
    # we need to multiply by vt_rot.T
    s_rot = u.T.dot(sim_mat).dot(vt_rot.T)

    # Return the eigenvalues of the rotated matrix according to
    # the fact that the eigenvalues output by PCA are:
    # eigvals = Sigma^2
    eigvals_rot = np.diag(s_rot) ** 2
    eigvals_rot_sorted = np.sort(eigvals_rot)
    eigvals_rot_sorted_descending = eigvals_rot_sorted[::-1]

    # Determine the number of components to consider, by
    # fitting a power curve to the first 50 'rotate' eigenvalues
    npc = mt.fit_power(eigvals_rot_sorted_descending)

    print("We found " + str(npc) + " principal components")

    # Perform the second decomposition, with a given number of components
    pca = decomposition.PCA(n_components = npc)
    pca.fit(sim_mat)

    # Principal componets loadings (pc) and eigenvalues (eig_val)
    pc = pca.components_
    eig_val = pca.explained_variance_

    # Perform rotation of the eigenvectors/loadings
    # We need a matrix of pc in columns for the rotation, so we
    # take the transpose of the pca.components_ matrix
    pc_rot = mt.rotate_components(pc.T, gamma = 0.0)


    # Sort the clusters of ROIs and count the number of ROIs per clusters
    abs_v = np.abs(pc_rot)
    # plt.imshow(PCrot, aspect='auto');
    # plt.show()

    # Find the maximum loading for each ROI across principal components.
    # In other words, find to which principal component / cluster each ROI
    # should be assigned
    labels = np.argmax(abs_v, axis=1)

    ROI_clu = np.zeros(pc_rot.shape)

    # Display the assignment of each ROIs to its pc/cluster
    for i in np.arange(npc):
        factor_idx = np.where(labels == i)
        ROI_clu[factor_idx, i] = 1

    np.save(os.path.join(output_folder, "ROI_clu.npy"), ROI_clu)

    # 'labels' is already the vector of cluster number for each ROI
    # Here we just visualize it as in SPSS
    ROI_clu_sort = np.zeros(ROI_clu.shape)

    start_row = 0

    for i in np.arange(ROI_clu_sort.shape[1]):
        tmp = np.where(ROI_clu[:,i])
        nROIs_ith_clu = np.array(tmp).shape[1]
        stop_row = start_row + nROIs_ith_clu
        ROI_clu_sort[start_row : (nROIs_ith_clu + start_row),i] = 1
        start_row = stop_row

    # Show how clusters look like in the similarity_matrix
    idx_sort = np.argsort(labels)
    sim_mat_clusters = sim_mat[idx_sort,:][:,idx_sort]

    np.save(os.path.join(output_folder, "sim_mat_clusters.npy"),
            sim_mat_clusters)

    return labels
