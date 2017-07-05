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
        s_m = sim_mat + 0
        sim_mat = s_m
        labels = KMeans(n_clusters=nb_clu, n_init=20).fit_predict(sim_mat)

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
        sim_mat = similarity_matrix + 0

        # Get the eigenvalues and eigenvectors of the
        # sim_mat = cov(2D_connectivity_matrix)
        gamma_eigval, omega_eigvec = np.linalg.eig(sim_mat)

        # Sort the Gamma_eigval in decreasing order of magnitude, and sort
        # the order of the eigenvectors accordingly
        indsort = np.argsort(gamma_eigval)[::-1]

        # The SSQ_loadings is equal to the eigenvalues of the SM (cov(data))
        # They correspond to the values in the 'Extraction Sum of Squared
        # loadings' in SPSS
        gamma_eigval_sort = gamma_eigval[indsort]
        omega_eigvec_sort = omega_eigvec[:,indsort]
        SSQ_loadings = gamma_eigval_sort

        # Force all eigenvalues to be > 0, as rounding errors can yield very
        # small eigenvalues to be < 0
        # https://de.mathworks.com/matlabcentral/newsreader/view_thread/322098
        ind_negative_eigenvalues = np.where(gamma_eigval_sort < 0)
        gamma_eigval_sort[ind_negative_eigenvalues] = np.abs(
            gamma_eigval_sort[ind_negative_eigenvalues]);



        # --------------------------------------------------------------------------
        # (2) Calculate Factor loadings Lambda = Eigvecs * sqrt(Eigvals)
        # These correspond to the values in the 'Component Matrix' of SPSS
        # --------------------------------------------------------------------------
        Lambda = omega_eigvec_sort.dot( np.diag(gamma_eigval_sort**(1/2)) )



        # --------------------------------------------------------------------------
        # (3-4) Perform Orthomax rotation of Lambda -> Lambda_rot
        # Returns the Lambda_rot and its Sum of Squared loadings
        # This correspond to the values in the 'Rotated Component Matrix' of SPSS
        # up to a constant rotation factor.
        # --------------------------------------------------------------------------
        # Rotate factor loadings using the function in do_PCA_utilities.py
        # https://en.wikipedia.org/wiki/Talk:Varimax_rotation
        lambda_rot = mt.rotate_components(Lambda, gamma=rotation)

        # Get sum of squared loadings
        SSQ_loadings_rot = np.sum(lambda_rot**2, axis=0)
        # print(np.sort(SSQ_loadings_rot)[::-1])

        # Sort the SSQ_loadings_rot in descending order to prepare for the
        # power fitting
        SSQ_loadings_rot_sorted = np.sort(SSQ_loadings_rot)
        SSQ_loadings_rot_sorted_descending = SSQ_loadings_rot_sorted[::-1]
        # plt.plot(SSQ_loadings_rot_sorted_descending); plt.show()



        # --------------------------------------------------------------------------
        # (5) Fit a power law to the sorted SSQ_Loadings_rot to Estimate
        #     the number of relevant factors Npc using the fitpower function in
        #     do_PCA_utilities.py (only the first 50 SSQ_Loadings are considered).
        # Returns the number of components to consider: Npc
        # --------------------------------------------------------------------------
        npc = mt.fit_power(SSQ_loadings_rot_sorted_descending)
        print('\n Power fitting of the eigenvalues associated \
              with the rotated loadings')
        print('estimated the presence of ' + str(npc) + ' clusters \n')



        # --------------------------------------------------------------------------
        # (6) Rotate Lambda_Npc = Lambda[:,Npc]
        # Returns the final Factor loadings, defining the clusters
        # --------------------------------------------------------------------------
        lambda_npc = Lambda[:, 0:npc]
        lambda_npc_rot = mt.rotate_components(lambda_npc, gamma=rotation)




        # --------------------------------------------------------------------------
        # (7) Sort for visualization and return cluster labels
        # --------------------------------------------------------------------------
        # Sort the clusters of ROIs and count the number of ROIs per clusters
        absV = np.abs(lambda_npc_rot)
        # plt.imshow(PC, aspect='auto');
        # plt.show()

        # Find the maximum value for each ROI across principal components.
        # In other words, find to which principal component / cluster each ROI
        # should be assigned
        labels = np.argmax(absV,axis=1)

        ROI_clu = np.zeros(lambda_npc_rot.shape)

        # Display the assignment of each ROIs to its pc/cluster
        for i in np.arange(npc):
            factor_idx = np.where(labels == i)
            ROI_clu[factor_idx, i] = 1

        # plt.imshow(ROI_clu, aspect='auto', interpolation='none');
        # plt.show()

        # 'labels' is already the vector of cluster number for each ROI
        # Here we just visualize it as in SPSS
        ROI_clu_sort = np.zeros(ROI_clu.shape)

        startrow = 0

        for i in np.arange(ROI_clu_sort.shape[1]):
            tmp = np.where(ROI_clu[:,i])
            nROIs_ith_clu = np.array(tmp).shape[1]
            stoprow = startrow + nROIs_ith_clu
            ROI_clu_sort[startrow : (nROIs_ith_clu + startrow),i] = 1
            startrow = stoprow

        # plt.imshow(ROI_clu_sort, aspect='auto', interpolation='none');
        # plt.show()
        np.save(path_pref + 'ROI_clu.npy', ROI_clu)
        np.save(path_pref + "ROI_clu_sort.npy", ROI_clu_sort)

        # Show how clusters look like in the similarity_matrix
        idxsort = np.argsort(labels)
        sim_mat_clusters = sim_mat[idxsort,:][:,idxsort]

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


    # To label each voxel with the corresponding cluster value we need to:
    # (1) retrieve the voxel index (2D matrix row) for each ROI
    # (2) assign the same cluster value for all voxels in an ROI
    for ith_ROI in np.arange(len(labels)):
        ind_ith_clu = np.array(np.where(ROIs_labels == ith_ROI))  # (1)
        ind_clusters[ind_ith_clu] = labels[ith_ROI] + 1 # (2)


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
