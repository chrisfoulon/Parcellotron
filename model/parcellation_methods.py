# -*- coding: utf-8 -*-
import os
import numpy as np

from sklearn.cluster import KMeans
from sklearn import decomposition

import model.matrix_transformations as mt
import nibabel as nib


class Parceller():
    handled_methods = ['PCA', 'KMeans']
    rotation_PCA = ['quartimax', 'varimax']
    def __init__(self, method, sim_mat, out_path, param):
        assert method in Parceller.handled_methods,\
               "The method: " + method + " is not handled"
        self.method = method
        self.out_path = out_path
        if method == "PCA":
            assert param in Parceller.rotation_PCA,\
                "Wrong rotation parameter: " + str(param)
            self.temp_files = [self.out_path + "ROI_clu.npy",
                               self.out_path + "ROI_clu_sort.npy"]
            self.param = param
            self.labels = self.parcellate_PCA(sim_mat, self.out_path, self.param)

        if method == "KMeans":
            assert type(param) is int, "The number of cluster must be an int"
            self.temp_files = [self.out_path + "sim_mat_KMeans.npy"]
            self.param = param
            self.labels = self.parcellate_KMeans(sim_mat, self.out_path, self.param)

    def parcellate_KMeans(self, sim_mat, path_pref, nb_clu):
        """ Parellate a 2D similarity matrix with the KMeans algorithm
        Parameters
        ----------
        sim_mat: 2D np.array
            square similarity_matrix, e.g. correlation matrix
        nb_clu: int
            desired number of clusters
        path_pref: str
            the path and the prefixes to add before the name of files which will be
            created by the function
        Returns
        -------
        labels: np.array
            Nseed labels (integers) which can be used to assign to each seed
            ROI the value associated to a certain cluster
        """
        labels = KMeans(n_clusters=nb_clu).fit_predict(sim_mat)

        IDX_CLU = np.argsort(labels)

        similarity_matrix_reordered = sim_mat[IDX_CLU,:][:,IDX_CLU]

        np.save(path_pref + "sim_mat_KMeans.npy", similarity_matrix_reordered)

        return labels

    def parcellate_PCA(self, similarity_matrix, path_pref, rot='quartimax'):
        """ Parellate a 2D similarity matrix with the PCA algorithm
        Parameters
        ----------
        similarity_matrix: 2D np.array
            square similarity_matrix, e.g. correlation matrix
            (It is assumed that the original 2D_connectivity_matrix was
            normalized across ROIs)
        rot: str ['quartimax', 'varimax']
            Type of factor rotation
        path_pref: str
            the path and the prefixes to add before the name of files which will be
            created by the function
        Returns
        -------
        labels: np.array
            Nseed labels (integers) which can be used to assign to each seed
            ROI the value associated to a certain cluster
        """
        if rot == 'quartimax':
            rotation = 0.0
        elif rot == 'varimax':
            rotation = 1.0
        else:
            raise Exception('This factor rotation type is not handled')
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
        pc_rot = mt.rotate_components(pc, gamma = rotation)

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
        pc_rot = mt.rotate_components(pc.T, gamma = rotation)


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

        np.save(path_pref + 'ROI_clu.npy', ROI_clu)

        # 'labels' is already the vector of cluster number for each ROI
        # Here we just visualize it as in SPSS
        ROI_clu_sort = np.zeros(ROI_clu.shape)

        start_row = 0

        for i in np.arange(ROI_clu_sort.shape[1]):
            tmp = np.where(ROI_clu[:,i])
            nROIs_ith_clu = np.array(tmp).shape[1]
            stop_row = start_row + nROIs_ith_clu
            ROI_clu_sort[start_row: (nROIs_ith_clu + start_row),i] = 1
            start_row = stop_row

        # Show how clusters look like in the similarity_matrix
        idx_sort = np.argsort(labels)
        sim_mat_clusters = sim_mat[idx_sort,:][:,idx_sort]

        np.save(path_pref + "ROI_clu_sort.npy", ROI_clu_sort)

        return labels

def write_clusters(shape, affine, ROIs_labels, labels, seed_coord, res_dir,
                   out_pref):
    """ Write the clusters in a nifti 3D image where the voxels values are
    the cluster labels
    """
    # Create an empty volume to store the clusters
    nii_mask = np.zeros(shape)
    nvox = len(ROIs_labels)
    # prepare a vector of length nvox-in-seed = len(ROIlabels), to store
    # the cluster label for each voxel of the seed region
    ind_clusters = np.zeros(nvox)
    import view.display_intermediates as di


    # To label each voxel with the corresponding cluster value we need to:
    # (1) retrieve the voxel index (2D matrix row) for each ROI
    # (2) assign the same cluster value for all voxels in an ROI
    for ith_ROI in np.arange(len(labels)):
        ind_ith_clu = np.array(np.where(ROIs_labels == ith_ROI))  # (1)
        ind_clusters[ind_ith_clu] = labels[ith_ROI] + 1 # (2)

    print("ind_cluster shape: " + str(ind_clusters.shape))
    # We take the vector ind_clusters containing the cluster values
    # for each voxel and we assign that value in the corresponding xyz
    # coordinates
    for jth_vox in np.arange(nvox):
        vox = seed_coord[jth_vox,:].astype('int64')
        nii_mask[vox[0], vox[1], vox[2]] = ind_clusters[jth_vox]


    nii_cluster = nib.Nifti1Image(nii_mask, affine)

    clusters_img = os.path.join(res_dir, out_pref + 'clusters.nii.gz')
    nib.save(nii_cluster, clusters_img)

    # We return the path to the nifti file
    return clusters_img
