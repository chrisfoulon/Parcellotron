# -*- coding: utf-8 -*-

import os
import glob
import shutil
import textwrap
import abc
import numpy as np
import collections
import math

import pandas as pd
import nibabel as nib
from sklearn.cluster import KMeans
from sklearn import decomposition
from scipy import sparse

import model.utils as ut
import model.matrix_transformations as mt
import model.similarity_matrices as sm
import model.parcellation_methods as pm


class Parcellobject(metaclass=abc.ABCMeta):
    """ @Inheritance Parcellobject:
    Abstract class of an generic object which will contain informations
    used to parcellate an image.
    Parameters
    ----------
    subj_path: str
        The path to the folder containing the different modality folders which
        contain the inputs.
    modality: str
        the name of the modality, it's automatically handled by the subclasses
    group_level: bool
        true if you want to do group level analysis
    seed_pref: str [optional]
        prefix of the seed file
    target_pref: str [optional]
        prefix of the target file

    Attributes
    ----------
    modality: {'Tracto_4D', 'Tracto_mat', 'FMRI_4D'}
        Name of the modality
    seed_pref: str [optional]
        prefix of the seed file
    target_pref: str [optional]
        prefix of the target file
    out_pref: str
        prefix used to defferentiate the outputs (temporary or not) between
        analyses with different seeds and/or targets
    subj_folder: str
        Path to the subject folder
    subj_name: str
        Name of the subject
    input_dir: str
        Directory containing the inputs of the subject for this modality
    root_dir: str
        Parent directory of the subject folder (used especially for group level
        analysis)
    group_level: bool (default: False)
        If True, the seed and target files will be searched and created in a
        general folder in the root directory.
        If False, all the files are searched and created in the subject folder
    res_dir: str
        Folder which will contain the different results of the software, for
        this modality.  If it does not existThe folder is created when the
        object is instanciated
    in_dict: dict
        A dictionnary storing the key string to find in input files and their
        corresponding input files.
        This attribut have to be filled by self.verify_input_folder()
    seed_path: str
        Path to the seedROIs nifti file which is an image with values indexing
        Regions of Interest (ROIs)
    target_path: str
        Path to the nifti 3D image of the target mask
    seed_target_folder: str
        Folder which contains the seed and the target files
    seed_coord: np.array
        Index of the xyz coordinates of the voxels in each ROIs
    ROIs_label: np.array
        Create an index of the ROI associated with each row of the
        (subsequently) created 2D_connectivity_matrix
    cmap2D_path: str
        Path to the python file where the connectivity matrix is or will be
        stored. If the matrix already exists, we won't compute it again
    co_mat_2D: np.array
        The 2D connectivity matrix calculated from the data
    final_shape: array
        The shape of the seedROIs file
    final_affine

    @Inheritance END


    """

    software_name = "Parcellotron"
    seed_name = "seedROIs"
    target_name = "targetMask"

    @abc.abstractmethod
    def __init__(self, subj_path, modality, group_level=False, seed_pref='',
                 target_pref=''):
        self.__doc__ = Parcellobject.__doc__ + "####  " + \
            self.__class__.__name__ + "  ####\n     " + self.__doc__
        self.modality = modality
        self.seed_pref = ut.format_pref(seed_pref)
        self.target_pref = ut.format_pref(target_pref)
        self.out_pref = self.seed_pref + self.target_pref
        self.subj_folder = subj_path
        self.subj_name = os.path.basename(self.subj_folder)
        self.input_dir = os.path.join(self.subj_folder, modality)
        self.root_dir = ut.parent_dir(self.subj_folder)
        self.group_level = group_level

        self.res_dir = os.path.join(self.input_dir, "_" +
                                    Parcellobject.software_name + "_results")
        if not os.path.exists(self.res_dir):
            print("The result directory is: " + self.res_dir)
            os.mkdir(self.res_dir)
            print("And it is supposed to be created here.")

        # self.temp_dict = self.init_temp_paths(self.res_dir)

        # We check if the input files needed exist and we store their paths
        self.in_dict = self.init_input_dict()
        assert self.verify_input_folder(self.input_dir), self.inputs_needed()

        # Here we check if we do group level analysis or not
        if self.group_level:
            self.seed_target_folder = os.path.join(
                self.root_dir, "_group_level")
            assert os.path.exists(self.seed_target_folder), """The general
            folder "_group_level" does not exist
            """
        else:
            self.seed_target_folder = self.input_dir
            assert os.path.exists(self.seed_target_folder), """The input
            folder does not exist
            """

        self.target_path = ut.find_with_pref(
            self.seed_target_folder, self.seed_pref, Parcellobject.target_name)

        # Create a map of correspondence among ROIs and voxels, where the ROI
        # order also reflects that of the (subsequent) rows of the connectivity
        # matrix
        (self.seed_coord, self.ROIs_labels) = self.map_ROIs()
        # Name of the 2D connectivity matrix file
        self.cmap2D_path = os.path.join(
            self.res_dir, self.subj_name + "_cmap2D.npy")

        if os.path.exists(self.cmap2D_path):
            self.co_mat_2D = np.load(self.cmap2D_path)
        else:
            # Create the 2D connectivity matrix if it does not exist
            self.co_mat_2D = self.read_inputs_into_2D()
            # Save the connectivity_matrix in an npy file
            np.save(self.cmap2D_path, self.co_mat_2D)

        self.set_final_shape()

    # Init functions
    @abc.abstractmethod
    def init_input_dict(self):
        """ Fill input files substring to check and store input files paths.
        The function have to return the dictionnary
        """
        pass

    @abc.abstractmethod
    def inputs_needed(self):
        """ Display a message to explain what input files you need and which
        name you need to indentify those files.
        """

        return textwrap.dedent("""
            The seed and target files should be in the subject input folder
            of in a folder called "group_level" in the root folder (so the
            parent folder of the subject directory)
            +)  [target_prefix_]targetMask.nii[.gz]: 3D binary mask of the
                target area
            ++) [seed_prefix_]seedROIs.nii[.gz]: 3D file with values indexing
                Regions of Interest (ROIs).
            """)

    def verify_input_folder(self, in_path):
        """ This function aims to fill self.in_dict and verify that all the
        input files need are in self.input_folder.
        Note that the input files are only the files specific for a particular
        modality type. The target and seed files will be handled in another
        function.
        """
        assert hasattr(self, 'in_dict'), "self.in_dict is not initialized"
        assert self.in_dict != {}, "self.in_dict is empty !"
        boo = True
        for k in self.in_dict.keys():
            res = ut.find_in_filename(in_path, k)
            if res == "":
                print('I did not find the ' + k + ' file.')
            self.in_dict[k] = res
            boo = boo and (res != "")
        print("The dictionary which is storing the inputs: ")
        print(self.in_dict)
        return boo

    def init_temp_paths(self, path):
        """ Initialize the paths of temprorary_files and return a dictionary
        with the paths associated with simple keys.

        Parameters
        ----------
        path: str
            the path of the folder which will contain the temprorary_files

        Returns
        -------
        dd: dict
            simple key associated with the path to the file
        """
        dd = {'sim_mat_KMeans': os.path.join(
                  path, self.out_pref + 'sim_mat_KMeans.npy'),
              "ROI_clu_sort": os.path.join(
                  path, self.out_pref + '"ROI_clu_sort".npy'),
              'ROI_clu': os.path.join(
                  path, self.out_pref + 'ROI_clu.npy')}
        return dd

    def set_final_shape(self):
        """ Load the seedROIs file to store its shape and affine in object
        attributes
        """
        nii = nib.load(self.seed_path)
        self.final_shape = nii.shape
        self.final_affine = nii.affine

    # Tools
    def reset_outputs(self):
        """ This function will remove the content of self.res_dir
        """
        shutil.rmtree(self.res_dir)

    # Calculation functions
    @abc.abstractmethod
    def read_inputs_into_2D(self):
        pass

    @abc.abstractmethod
    def map_ROIs(self):
        self.seed_path = ut.find_with_pref(
            self.seed_target_folder, self.seed_pref, Parcellobject.seed_name)

    def mat_transform(self, option, mat):
        """ Calculate the chosen transformation on the matrix given
        Parameters
        ----------
        option: str ['log2', 'log2_zscore', 'zscore', 'none']
            The name of the parcellation method
        mat: np.array
        """
        tr_mat_2D = mat
        if option in ['log2', 'log2_zscore']:
            tr_mat_2D = mt.matrix_log2(self.co_mat_2D)
        if option in ['zscore', 'log2_zscore']:
            tr_mat_2D = mt.matrix_zscore(self.co_mat_2D)
        self.tr_mat_2D = tr_mat_2D

        return self.tr_mat_2D

    def similarity(self, option, mat):
        """ Calculate the similarity matrix of the matrix given in parameters
        Parameters
        ----------
        option: str ['covariance', 'correlation', 'distance']
            The name of the parcellation method
        mat: np.array
        """
        if option == 'covariance':
            sim_mat = sm.similarity_covariance(mat)
        if option == 'correlation':
            sim_mat = sm.similarity_correlation(mat)
        if option == 'distance':
            sim_mat = sm.similarity_distance(mat)
        self.sim_mat = sim_mat
        return self.sim_mat

    def parcellate(self, option, sim_mat, param_parcellate=None):
        """ Perform the parcellation chosen on the matrix.
        Parameters
        ---------
        option: str ['KMeans', 'PCA']
            The name of the parcellation method
        sim_mat: np.array
            The similarity matrix you want to parcellate
        KMeans_nclu: int [default: None]
            Number of clusters you want to find with the KMeans algorithm
        """
        pref = option
        self.out_pref += option + '_'
        if option == 'PCA':
            self.out_pref += param_parcellate + '_'
        path_pref = os.path.join(self.res_dir, self.out_pref)
        if param_parcellate != None:
            parceller = pm.Parceller(option, sim_mat, path_pref,
                                     param_parcellate)
            labels = parceller.labels

        self.labels = labels

        return self.labels

    def write_clusters(self):
        """ Write the clusters in a nifti 3D image where the voxels values are
        the cluster labels
        """
        self.clusters_path = pm.write_clusters(self.final_shape,
                                               self.final_affine,
                                               self.ROIs_labels,
                                               self.labels,
                                               self.seed_coord,
                                               self.res_dir,
                                               self.out_pref)


class Tracto_4D(Parcellobject):
    """ Object containing the informations used to parcellate the tractography
    of 1 subject from a 4D image.
    """
    def __init__(self, subj_path, group_level=False, seed_pref='',
                 target_pref=''):
        super().__init__(subj_path, self.__class__.__name__, group_level,
                         seed_pref, target_pref)

    def init_input_dict(self):
        """ Fill input files substring to check and store input files path
        Returns
        -------
        d: dict
            the dictionnary containing the substring to find the input files
        """
        d = {'cmaps4D':''}
        return d

    def inputs_needed(self):
        """
        Returns
        -------
        message: str
            A string describing the inputs you need for this modality
        """
        message = textwrap.dedent("""\
            Inputs needed for this modality:
            1) subj_4Dcmaps.nii[.gz] a 4D image with a connectivity map
                for each time point
            """) + super().inputs_needed()
        return message

    def read_inputs_into_2D(self):
        """ Read the inputs and transform the 4D image into a 2D connectivity
        matrix.
        Returns
        -------
        co_mat_2D: 2D np.array
            2D matrix where rows are seed ROIs and columns the target's voxels

        Notes
        -----
        co_mat_2D is also stored in "temprorary_files" in the results folder
        with the name: subj_2D_connectivity_matrix.npy
        """
        # Read the brain ribbon mask, which will become the number of columns
        # of the 2D connectivity matrix
        ribbon_data = nib.load(self.target_path).get_data()

        # Create indices for voxels on the brain ribbon
        ind_ribbon = np.where(ribbon_data)
        nvox = np.array(ind_ribbon).shape[1]

        # Now we load the 4D file with the connectivity profiles for each ROI
        co = nib.load(self.in_dict['cmaps4D']).get_data()

        # Record the number of ROIs
        nROIs = co.shape[3]

        # Prepare a zero matrix to store the 2D connectivity matrix
        co_mat_2D = np.zeros((nROIs,nvox))

        # Fill the connectivity matrix
        for i in np.arange(nROIs):
            tmp = co[:,:,:,i]
            co_mat_2D[i,:] = tmp[ind_ribbon]

        return co_mat_2D

    def map_ROIs(self):
        super().map_ROIs()
        return ut.read_ROIs_from_nifti(self.seed_path)


class Tracto_mat(Parcellobject):
    """ Description
    Parameters
    ----------
    ROI_size: int
        The ROIs' size you want for the seed region

    Attributes
    ----------
    seed_mask: str
        path to the mask of the seed (nifti file)
    ROI_size: int
        The ROIs' size
    in_names: dict
        Associate easier keys to the in_dict keys.
        For instance: self.in_dict[self.in_names['fdt_matrix']]
    """
    def __init__(self, subj_path, ROIs_size, group_level=False, seed_pref='',
                 target_pref=''):
        self.ROIs_size = ROIs_size
        super().__init__(subj_path, self.__class__.__name__, group_level,
                         seed_pref,
                         target_pref)


    def init_input_dict(self):
        """ Fill input files substring to check and store input files path
        Returns
        -------
        d: dict
            the dictionnary containing the substring to find the input files
        """
        # to simplify a bit the access to the elements
        self.in_names = {
            'fdt_coord':os.path.join('omat*', 'coords_for_fdt_matrix'),
            'fdt_matrix':os.path.join('omat*', 'fdt_matrix*.dot'),
            'fdt_paths':os.path.join('omat*', 'fdt_paths.nii.gz')}
        d = {self.in_names['fdt_coord']:'',
             self.in_names['fdt_matrix']:'',
             self.in_names['fdt_paths']:''}
        return d

    def inputs_needed(self):
        """
        Returns
        -------
        message: str
            A string describing the inputs you need for this modality
        """
        message = textwrap.dedent("""\
            Inputs needed for this modality:
            1) coord_for_fdt_matrix[1-2-3].nii[.gz], fdt_paths.nii[.gz],
                fdt_matrix[1-2-3].dot are the outputs of probtrackx software
            """) + super().inputs_needed()
        return message

    def read_inputs_into_2D(self):
        """ Read the outputs of probtrackx and transform into a 2D connectivity
        matrix.
        Returns
        -------
        co_mat_2D: 2D np.array
            2D matrix where rows are seed ROIs and columns the target's voxels

        Notes
        -----
        co_mat_2D is also stored in "temprorary_files" in the results folder
        with the name: subj_2D_connectivity_matrix.npy
        """
        # Convert the omat3/fdt_matrix3.dot (whole brain) into npy
        fdt_matrix = self.convert_dotbigmat()
        target_ind, _ = self.get_mask_indices(self.target_path)
        print("I will create the ROIzed connectivity matrix")
        print("Time for a coffee...")

        # (1)  Import the sparse matrix in binary Python format
        origmat_3cols = np.load(self.fdt_matrix_py_file)
        print(origmat_3cols.shape)
        print("Shape seed")


        # (2) Need to subtract 1 since in Python the subscripts start from 0
        rows = origmat_3cols[:,0].astype('int') - 1
        cols = origmat_3cols[:,1].astype('int') - 1
        data = origmat_3cols[:,2].astype('int32')
        del origmat_3cols

        origmat = sparse.csr_matrix((data, (rows, cols)))
        del rows, cols, data

        if np.sum(sparse.tril(origmat, -1)) == 0:
            print("I found omat3 format")
            origmat = origmat + sparse.triu(origmat, 1).T
        else:
            print("I found omat1 format")

        # # (3) Create the sparse matrix and add the lower triangular
        # origmat_triu = sparse.csr_matrix((data, (rows, cols)))
        # del rows, cols, data
        # print(origmat_triu.shape)
        # # For omat1, the matrix is full and NOT symmetric, therefore
        # # we should NOT add the transposed of the upper triangular
        # origmat = origmat_triu
        # del origmat_triu
        #
        # # # For omat3, the matrix is only upper triangular, therefore we Need
        # # # to transpose and add
        # # origmat = origmat_triu + sparse.triu(origmat_triu,1).T
        # # del origmat_triu
        #
        # print(origmat.shape)
        # (4) Convert to np.array for ROIzation
        # origmat = origmat.astype('int32')
        # origmat = origmat.toarray()

        # origmat[0:3,0:3]
        # import matplotlib.pyplot as plt
        # plt.imshow(origmat);
        # plt.show()



        # (5) Sum over the rows of origmat to group the connectivity profiles by ROI
        nROIs = len(np.unique(self.ROIs_labels))
        connectivity_matrix = np.zeros((nROIs, origmat.shape[1] ))

        for ith_ROI in np.arange(nROIs):
            # find the indices of seed_coord pointing to
            #                                         the voxels inside the ROI
            ind_ith_ROI_in_seed_coord = np.squeeze(
                np.array(np.where(self.ROIs_labels == ith_ROI)))

            # find the index of origmat/coord pointing to
            #                 the indices of seed coord pointing to
            #                                         the voxels inside the ROI
            ind_ith_ROI_in_origmat = self.seed_ind[ind_ith_ROI_in_seed_coord]

            # do the sum of the rows of origmat corresponding to the voxels in each ROI
            connectivity_matrix[ith_ROI,:] = np.sum(
                origmat[ind_ith_ROI_in_origmat,:], axis=0 )



        # (6) Remove origmat columns (i.e. ribbon voxels) which are not in the target
        dim_ribbon = origmat.shape[1]
        if dim_ribbon != len(target_ind):
            connectivity_matrix = connectivity_matrix[:,target_ind]



        # (7) Save the ROIzed connectivity_matrix in an npy file
        np.save(self.cmap2D_path, connectivity_matrix)

        # # Display the ROIzed connectivity matrix
        # import matplotlib.pyplot as plt
        # plt.imshow(np.log2(connectivity_matrix+1), interpolation='none', aspect='auto')
        # plt.show()
        return connectivity_matrix

    def map_ROIs(self):
        # seed_mask will be used to create the seedROIs. This file can be
        # in the subject input folder or in the general group input folder
        self.seed_mask = ut.find_with_pref(
            self.seed_target_folder, self.seed_pref, 'seedMask')
        # We have to handle the ROIs_size more elegantly ! But later, today
        # I have swimming pool.
        if self.ROIs_size == 0:
            data = nib.load(self.seed_mask).get_data()
            w = np.where(data)
            # There is probably a more elegant way but it seems to work
            size = len(w[0])
            # So the default ROIs_size is the size of the seed / 100
            self.ROIs_size = int(math.floor(size / 100))
        self.out_pref += str(self.ROIs_size) + "_"
        self.seed_path = os.path.join(self.seed_target_folder,
                                 self.seed_pref + str(self.ROIs_size) + \
                                 "_seedROIs.nii.gz")
        # We need self.seed_ind for the creation of the 2D connectivity matrix
        self.seed_ind, self.seed_coord = self.get_mask_indices(self.seed_mask)

        print("Seed coord shape after get mask")
        print(self.seed_coord.shape)
        if os.path.exists(self.seed_path):
            return ut.read_ROIs_from_nifti(self.seed_path)
        else:
            nii = nib.load(self.in_dict[self.in_names['fdt_paths']])
            nii_dims = nii.header.get_zooms()

            voxel_volume = np.prod(nii_dims)
            # voxel_volume = np.prod(np.diag(img.affine))  # alternative method
            ribbon_volume = len(self.seed_coord) * voxel_volume
            # number of clusters to create
            k = int(math.floor(ribbon_volume / self.ROIs_size))

            print('There are ' + str(np.shape(self.seed_coord)[0]) +
                  ' voxel for a total of ~' + str(ribbon_volume.astype('int')) +
                  ' mm3')
            print('One voxel is ' + str(round(voxel_volume,3)) + ' mm3')
            print('With the chosen ROI size of ' + str(self.ROIs_size) +
                  ' mm3 there will be ' + str(k) + ' ROIs')
            print(' ')
            print('I need to create the seed ROIs.')
            print('This might take some time for large seed regions...')

            # t0 = time.time()
            ROIlabels = KMeans(n_clusters=k, n_init=10).fit_predict(
                self.seed_coord)
            # t1 = time.time()
            # print("kmeans performed in %.3f s \n" % (t1 - t0))

            # Create a nifti file with the ROIs

            # min(ROIlabels)
            # max(ROIlabels)
            # img_ROIs_filename = os.path.join(
            #     self.seed_target_folder,
            #     self.seed_pref + "_" + self.ROIs_size + "_seedROIs.npz"
            # )
            # np.savez(ROIfile, ROIlabels=ROIlabels)

            mask = np.zeros(nii.get_data().shape)
            for i in np.arange(k):
                ind = np.where(ROIlabels==i)
                mask[self.seed_coord[ind,0],
                     self.seed_coord[ind,1], self.seed_coord[ind,2]] = i + 1
            # To be in the same orientation as the 4D method
            print("seed_path: ")
            print(self.seed_path)
            img_ROIs = nib.Nifti1Image(mask, nii.affine)

            nib.save(img_ROIs, self.seed_path)


            print('I created' + self.seed_path + ' for you');
            print('The ROIlabels are ordered as the rows of seed_coord')
            return self.seed_coord, ROIlabels


        # Read seed_mask, target_mask and coord_for_fdt_matrix3
        # seed_ind, seed_coord = get_mask_indices(bd,subj,hemi,'seed_mask')
        # target_ind, _ = get_mask_indices(bd,subj,hemi,'target_mask')
        # CREATE seedROIs if it does not exist

    def convert_dotbigmat(self):
        """ Import the file fdt_matrix.npy into a np.array. If the file does
        not exist, the function will import the fdt_matrix.dot file, save it
        into the .npy file and return the np.array of its content.
        Returns
        -------
        fdt_matrix: np.array
            The raw connectivity matrix in a python format.
            The array contains 3 columns: x, y, value
        """
        fdt_dotmatrix_file = self.in_dict[self.in_names['fdt_matrix']]
        print("ST_folder:" + self.seed_target_folder)
        self.fdt_matrix_py_file = os.path.join(
            self.res_dir, 'fdt_matrix.npy')

        if os.path.exists(self.fdt_matrix_py_file):
            fdt_matrix = np.load(self.fdt_matrix_py_file)
        else:
            print("Please wait while I convert the fdt_matrix3.dot into \
                  Python format...")
            # To know how much time it takes

            fdt_dotmatrix_df = pd.read_csv(fdt_dotmatrix_file,
                                           header=None, delim_whitespace=True)
            fdt_dotmatrix = fdt_dotmatrix_df.as_matrix()

            # save the matrix in binary format
            np.save(self.fdt_matrix_py_file, fdt_dotmatrix)

            fdt_matrix = fdt_dotmatrix

        return fdt_matrix

    def get_mask_indices(self, mask_path):
        """ Retrieve the indices of the mask inside the coordinates matrix of
        probtrackx
        Parameters
        ----------
        mask_path: str
            Path to the mask in nifti
        Returns
        -------
        ind_mask:
        coord_mask:
        """
        # Load the coordinates of the voxels on the whole-brain ribbon.
        # The order of the coordinates in this text file is the same as the
        # rows in the fdt_matrix3.dot matrix.
        # NB: In matlab we need to add 1 since the matrix indices start from 1.
        #     In Python they start from zero, so we leave them as they are.
        coord = np.genfromtxt(
            self.in_dict[self.in_names['fdt_coord']])[:,0:3].astype('int')

        # Read the [mask_path].nii.gz
        masknii = nib.load(mask_path).get_data()

        # Get the xyz coordinates of each voxel in the mask
        mask = np.array(np.where(masknii)).T
        Nvoxels_in_mask = mask.shape[0]


        # To get the coordinates of the mask, we calculate the intersection
        # between the set of coordinates in coord (i.e. whole brain/rows of the
        # fdt_matrix3) and the set of coordinates in mask. Steps are detailed
        # below.

        # (1) Transform the np.array of coordinates in a set, retaining the
        #  original order. For this we need collections.OrderedDict since 'set'.
        mask_set = collections.OrderedDict.fromkeys(
            tuple(vox) for vox in mask)
        coord_set = collections.OrderedDict.fromkeys(
            tuple(vox) for vox in coord)

        # (2) Create an "ordered dictionary" of coord
        coord_dict = collections.OrderedDict(
            (key,value) for value,key in enumerate(coord_set))

        # (3) Take the intersection of the mask and coord set. For a dataset of
        # ~63K coordinates. This returns a set of common xyz coordinates between
        #  coord and mask. This method is ~250times faster than the one
        # commented below, using original np.arrays
        common_voxels = set(mask_set).intersection(coord_set)


        mask_len = len(mask)
        ind_mask = np.zeros(mask_len).astype('int')
        for i in np.arange(mask_len):
            ind_mask[i] = coord_dict[tuple(mask[i,:])]

        coord_mask = coord[ind_mask,:]
        # If u want to test, use the following lines
        # coord[ind_mask,:]
        # mask
        return ind_mask, coord_mask.astype('int')


# %%
# test1 = Tracto_4D("/data/BCBLab/test_COBRA/S1")
# mat = test1.co_mat_2D
#
# mat.shape
# print(test1.read_inputs_into_2D.__doc__)
# ind, rows = test1.map_ROIs
# # %%
# st = os.path.join("blabla", "bliblibli")
# print("\n" + os.path.dirname(
#     os.path.dirname(os.path.join("/data/BCBLab/test_COBRA/S1/", ""))))
# # os.rmdir("blibli")
# print(st)
# tt = """test
# test bla"""
# type(st)
# def func(str):
#     assert len(str) > 2, "Error lol"
#     print(str)
# func("1")
# def returns_str():
#     return "une string"
# assert 2 == 1, returns_str()
