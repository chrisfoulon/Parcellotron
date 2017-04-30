# -*- coding: utf-8 -*-

import abc
import nibabel as nib
import numpy as np
import utils as ut
import os
import glob
import shutil
import textwrap

class Parcellotron(metaclass=abc.ABCMeta):
    """ @Inheritance Parcellotron:
    Abstract class of an generic object which will contain informations
    used to parcellate an image.
    Attributes
    ----------
    modality: {'tracto_4D', 'tracto_matrix', 'fmri_4D'}
        Name of the modality
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
        A dictionnarie storing the key string to find in input files and their
        corresponding input files.
        This attribut have to be filled by self.verify_input_folder()
    seed_path: str
        Path to the seedROIs nifti file which is an image with values indexing
        Regions of Interest (ROIs)
    target_path: str
        Path to the nifti 3D image of the target mask
    ind_xyz_ROIs : TO FILL
        Index of the xyz coordinates of the voxels in each ROIs
    ROIs_label : np.array
        Create an index of the ROI associated with each row of the
        (subsequently) created 2D_connectivity_matrix
    cmap2D_path : str
        Path to the python file where the connectivity matrix is or will be
        stored. If the matrix already exists, we won't compute it again
    co_mat_2D : np.array
        The 2D connectivity matrix calculated from the data
    """

    software_name = "COBRA"

    @abc.abstractmethod
    def __init__(self, subj_path, modality, group_level=False):
        self.__doc__ = Parcellotron.__doc__ + self.__doc__
        self.modality = modality
        self.subj_folder = subj_path
        self.subj_name = os.path.basename(subj_path)
        self.input_dir = os.path.join(subj_path, modality)
        self.root_dir = ut.parent_dir(subj_path)
        self.group_level = group_level

        self.res_dir = os.path.join(self.input_dir,
                                    Parcellotron.software_name + "_results")
        if not os.path.exists(self.res_dir):
            os.mkdir(self.res_dir)

        self.in_dict = self.init_input_dict()
        assert self.verify_input_folder(self.input_dir), self.inputs_needed()

        # Here we check if we do group level analysis or not
        if self.group_level:
            self.general_inputs = os.path.join(self.root_dir, "group_level")
            if not os.path.exists(self.general_inputs):
                os.mkdir(self.general_inputs)

            self.seed_path = os.join.path(self.general_inputs,
                                          "seedROIs.nii.gz")
            self.target_path = os.join.path(self.general_inputs,
                                            "targetMask.nii.gz")
        else:
            self.seed_path = os.path.join(self.input_dir, "seedROIs.nii.gz")
            self.target_path = os.path.join(self.input_dir, "targetMask.nii.gz")

        (self.ind_xyz_ROIs, self.ROIs_label) = self.map_ROIs()
        print("""ADD A PREFIX TO THE SEED (MAYBE ALSO TARGET) TO DISANTANGLE
              ANALYSES WITH DIFFERENT SEEDS AND IF THERE IS NO PREFIX
              WE JUST DO LIKE BEFORE""")
        # Name of the 2D connectivity matrix file
        self.cmap2D_path =\
            os.path.join(self.res_dir, self.subj_name + "_cmap2D.npy")

        if os.path.exists(self.cmap2D_path):
            self.co_mat_2D = np.load(self.cmap2D_path)
        else:
            # Create the 2D connectivity matrix if it does not exist
            self.co_mat_2D = self.read_inputs_into_2D()
            # Save the connectivity_matrix in an npy file
            np.save(self.cmap2D_path, self.co_mat_2D)
        # Create a map of correspondence among ROIs and voxels, where the ROI
        # order also reflects that of the (subsequent) rows of the connectivity
        # matrix
        ###### if (not MNISPACE_OPTION) and (not os.path.exists(MNIROIS_FILES)):
        (self.ind_xyz_ROIs, self.ROIs_label) = self.map_ROIs()

    # Init functions
    @abc.abstractmethod
    def init_input_dict(self):
        """ Fill input files substring to check and store input files path
        The function have to return the dictionnary
        """
        pass

    def verify_input_folder(self, in_path):
        """ This function aims to fill self.in_dict and verify that all the
        input files need are in self.input_folder
        """
        assert hasattr(self, 'in_dict'), "self.in_dict wasn't initialized"
        assert self.in_dict != {}, "self.in_dict is empty !"
        boo = True
        for k in self.in_dict.keys():
            res = ut.find_in_filename(in_path, k)
            if res == "":
                print('I did not find the ' + k + ' file.')
            self.in_dict[k] = res
            boo = boo and (res != "")
        return boo

    @abc.abstractmethod
    def inputs_needed(self):
        """ Display a message to explain what input files you need and which
        name you need to indentify those files.
        """
        pass

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

class tracto_4D(Parcellotron):
    """ Object containing the informations used to parcellate the tractography
    of 1 subject from a 4D image.
    Parameters
    ----------
    subj_path: str
        The path to the folder containing the different modality folders which
        contain the inputs.
    Attributes
    ----------
    INHERITED
    modality: {'tracto_4D', 'tracto_matrix', 'fmri_4D'}
        Name of the modality
    subj_folder: str
        Path to the subject folder
    subj_name: str
        Name of the subject
    root_dir: str
        Parent directory of the subject folder (used especially for group level
        analysis)
    input_dir: str
        Directory containing the inputs of the subject for this modality
    res_dir: str
        Folder which will contain the different results of the software, for
        this modality. The folder is created when the object is instanciate if
        it does not exist.
    """
    def __init__(self, subj_path):
        super().__init__(subj_path, self.__class__.__name__)

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
            2) subj_targetMask.nii[.gz] the 3D binary mask of the brain ribbon
            3) subj_seedROIs.nii[.gz] 3D file with values indexing
                Regions of Interest (ROIs).
            """)
        return message

    def read_inputs_into_2D(self):
        """ Read the inputs and tranform the 4D image into a 2D connectivity
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
        return ut.read_ROIs_from_nifti(self.seedROIs)

class tracto_mat(Parcellotron):
    """ Description
    """
    def __init__(self, subj_path):
        super().__init__(subj_path, self.__class__.__name__)
        # seed_mask will be used to create the seedROIs. This file can be
        # in the subject input folder or in the general group input folder
        self.seed_mask

    def init_input_dict(self):
        """ Fill input files substring to check and store input files path
        Returns
        -------
        d: dict
            the dictionnary containing the substring to find the input files
        """
        # self.in_names = {'coord':'omat*/coord_for_fdt_matrix'}
        d = {os.path.join('omat*', 'coord_for_fdt_matrix'):'',
             os.path.join('omat*', 'fdt_matrix':''),
             os.path.join('omat*', 'fdt_paths':'')}
        return d
        # self.in_dict[self.in_names['coord']]

    def inputs_needed(self):
        """
        Returns
        -------
        message: str
            A string describing the inputs you need for this modality
        """
        message = textwrap.dedent("""\
            Inputs needed for this modality:

            """)
        return message

    def read_inputs_into_2D(self):
        """ Read the inputs and tranform the 4D image into a 2D connectivity
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

    def map_ROIs(self):
        if os.path.exists(self.seedROIs):
            return ut.read_ROIs_from_nifti(self.seedROIs)
        # CREATE seedROIs if it does not exist


# %%
# test1 = tracto_4D("/data/BCBLab/test_COBRA/S1")
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
