# -*- coding: utf-8 -*-

import os
import glob
import shutil
import textwrap
import abc
import numpy as np
import collections

import pandas as pd
import nibabel as nib
from sklearn.cluster import KMeans
from sklearn import decomposition

import utils as ut
import matrix_transformations as mt
import similarity_matrices as sm
import parcellation_methods as pm


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
            os.mkdir(self.res_dir)

        self.temp_dict = self.init_temp_paths(self.res_dir)

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

        self.seed_path = ut.find_with_pref(
            self.seed_target_folder, self.seed_pref, Parcellobject.seed_name)
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
        self.inputs_needed.__doc__ =\
            cls.inputs_needed.__doc__ + self.inputs_needed.__doc__

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
        pass

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
        path_pref = os.path.join(self.res_dir, self.out_pref)
        print("ARGUMENTS : ")
        print(param_parcellate)
        if param_parcellate != None:
            parceller = pm.Parceller(option, sim_mat, path_pref,
                                     param_parcellate)
            labels = parceller.labels
        # if option == 'KMeans':
        #     if param_parcellate != None:
        #         labels = pm.parcellate_KMeans(sim_mat, param_parcellate,
        #                                       path_pref)
        #     else:
        #         raise Exception("""You need to define a number of cluster
        #                         to use the KMeans algorithm""")
        # elif option == 'PCA':
        #     if param_parcellate != None:
        #         labels = pm.parcellate_PCA(sim_mat, param_parcellate, path_pref)
        #     else:
        #         raise Exception("""You need to define the rotation you want
        #                         to to apply in the PCA algorithm""")
        # else:
        #     raise Exception(option + " is not yet implemented")

        self.labels = labels

        return self.labels

    def write_clusters(self, shape, affine, ROIs_labels, labels, seed_coord,
                       out_pref):
        """ Write the clusters in a nifti 3D image where the voxels values are
        the cluster labels
        """
        # Create an empty volume to store the clusters
        nii_mask = np.zeros(self.final_shape)

        nvox = len(self.ROIs_labels)
        # prepare a vector of length nvox-in-seed = len(ROIlabels), to store
        # the cluster label for each voxel of the seed region
        ind_clusters = np.zeros(nvox)



        # To label each voxel with the corresponding cluster value we need to:
        # (1) retrieve the voxel index (2D matrix row) for each ROI
        # (2) assign the same cluster value for all voxels in an ROI
        for ith_ROI in np.arange(len(self.labels)):
            ind_ith_clu = np.array(np.where(self.ROIs_labels == ith_ROI))  # (1)
            ind_clusters[ind_ith_clu] = self.labels[ith_ROI] + 1 # (2)

        # We take the vector ind_clusters containing the cluster values
        # for each voxel and we assign that value in the corresponding xyz
        # coordinates
        for jth_vox in np.arange(nvox):
            vox = self.seed_coord[jth_vox,:].astype('int')
            nii_mask[vox[0], vox[1], vox[2]] = ind_clusters[jth_vox]


        nii_cluster = nib.Nifti1Image(nii_mask, self.final_affine)

        self.clusters_img = os.path.join(self.out_pref + 'clusters.nii.gz')
        nib.save(nii_cluster, self.clusters_img)


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
        return ut.read_ROIs_from_nifti(self.seed_path)


class Tracto_mat(Parcellobject):
    """ Description
    Parameters
    ----------
    ROI_size: int
        The ROIs' size you want for the seed region

    Attributes
    ----------
    in_names: dict
        Associate easier keys to the in_dict keys.
        For instance: self.in_dict[self.in_names['fdt_matrix']]
    """
    def __init__(self, subj_path, ROIs_size, group_level=False, seed_pref='',
                 target_pref=''):
        super().__init__(subj_path, self.__class__.__name__, seed_pref,
                         target_pref)
        # seed_mask will be used to create the seedROIs. This file can be
        # in the subject input folder or in the general group input folder
        self.seed_mask = ut.find_with_pref(
            self.seed_target_folder, self.seed_pref, 'seedMask')
        self.ROIs_size = ROIs_size
        self.out_pref += str(self.ROIs_size) + "_"


    def init_input_dict(self):
        """ Fill input files substring to check and store input files path
        Returns
        -------
        d: dict
            the dictionnary containing the substring to find the input files
        """
        # to simplify a bit the access to the elements
        self.in_names = {
            'fdt_coord':os.path.join('omat*', 'coord_for_fdt_matrix'),
            'fdt_matrix':os.path.join('omat*', 'fdt_matrix'),
            'fdt_paths':os.path.join('omat*', 'fdt_paths')}
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
        fdt_matrix = convert_dotbigmat(bd,subj,hemi)

    def map_ROIs(self):
        if os.path.exists(self.seedROIs):
            return ut.read_ROIs_from_nifti(self.seedROIs)
        else:
            pass


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
        fdt_matrix_py_file = os.path.join(self.seed_target_folder,
                                          'fdt_matrix.npy')

        if os.path.exists(fdt_matrix_py_file):
            fdt_matrix = np.load(fdt_matrix_py_file)

        else:
            print("Please wait while I convert the fdt_matrix3.dot into \
                  Python format...")
            # To know how much time it takes

            fdt_dotmatrix_df = pd.read_csv(fdt_dotmatrix_file,
                                           header=None, delim_whitespace=True)
            fdt_dotmatrix = fdt_dotmatrix_df.as_matrix()
            # save the matrix in binary format
            np.save(fdt_matrix_py_file, fdt_dotmatrix)

            fdt_matrix = np.load(fdt_matrix_py_file)

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
        coord = np.genfromtxt(self.in_dict[self.in_names['fdt_coord']])[:,0:3]

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
