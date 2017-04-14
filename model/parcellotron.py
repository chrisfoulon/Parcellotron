# -*- coding: utf-8 -*-

import abc
import nibabel as nib
import numpy as np
import utils as ut
import os
import glob
import shutil
import textwrap

software_name = "COBRA"

class parcellotron(metaclass=abc.ABCMeta):
    """ Abstract class of an generic object which will contain informations
    used to parcellate an image.
    Attributes
    ----------
    modality : {'tracto_4D', 'tracto_matrix', 'fmri_4D'}
        Name of the modality
    subj_folder : str
        Path to the subject folder
    subj_name : str
        Name of the subject
    root_dir : str
        Parent directory of the subject folder (used especially for group level
        analysis)
    input_dir : str
        Directory containing the inputs of the subject for this modality
    in_dict : dict
        A dictionnarie storing the key string to find in input files and their
        corresponding input files.
        This attribut have to be filled by self.verify_input_folder()
    res_dir : str
        Folder which will contain the different results of the software, for
        this modality. The folder is created when the object is instanciate if
        it does not exist
    """
    @abc.abstractmethod
    def __init__(self, subj_path, modality):
        self.modality = modality
        self.subj_folder = subj_path
        self.subj_name = os.path.basename(subj_path)
        self.root_dir = os.path.dirname(subj_path) #Check here
        self.input_dir = os.path.join(subj_path, modality)
        self.res_dir = os.path.join(self.input_dir, software_name + "_results")
        if not os.path.exists(self.res_dir):
            os.mkdir(self.res_dir)
        self.in_dict = {}
        assert self.verify_input_folder(self.input_dir), self.inputs_needed()

    @abc.abstractmethod
    def read_inputs_into_2D(self):
        pass

    @abc.abstractmethod
    def map_ROIs(self):
        pass

    @abc.abstractmethod
    def verify_input_folder(self, path):
        """ This function aims to fill self.in_dict and verify that all the
        input files need are in self.input_folder
        """
        pass

    @abc.abstractmethod
    def inputs_needed(self):
        """ Display a message to explain what input files you need and which
        name you need to indentify those files.
        """
        pass

    def reset_outputs(self):
        """This function will remove the content of self.res_dir
        """
        shutil.rmtree(self.res_dir)


class tracto_4D(parcellotron):
    """ Object containing the informations used to parcellate the tractography
    of 1 subject from a 4D image.
    Parameters
    ----------
    subj_path : str
        The path to the folder containing the different modality folders which
        contain the inputs.
    Attributes
    ----------
    INHERITED
    modality : {'tracto_4D', 'tracto_matrix', 'fmri_4D'}
        Name of the modality
    subj_folder : str
        Path to the subject folder
    subj_name : str
        Name of the subject
    root_dir : str
        Parent directory of the subject folder (used especially for group level
        analysis)
    input_dir : str
        Directory containing the inputs of the subject for this modality
    res_dir : str
        Folder which will contain the different results of the software, for
        this modality. The folder is created when the object is instanciate if
        it does not exist.
    """
    def __init__(self, subj_path):
        super().__init__(subj_path, self.__class__.__name__)

    def verify_input_folder(self, in_path):
        boo = True
        self.in_dict = {'cmaps4D':'', 'seedROIs':'', 'targetRibbon':''}
        for k in self.in_dict.keys():
            res = ut.find_in_filename(in_path, k)
            if res == "":
                print('I did not find the ' + k + ' file.')
            self.in_dict[k] = res
            boo = boo and (res != "")
        return boo

    def inputs_needed(self):
        """
        Returns
        -------
        message : str
            A string describing the inputs you need for this modality
        """
        message = textwrap.dedent("""\
            Inputs needed for this modality :
            1) subj_4Dcmaps.nii[.gz] a 4D image with a connectivity map
                for each time point
            2) subj_ribbon.nii[.gz] the 3D binary mask of the brain ribbon
            3) subj_ROIs.nii[.gz] 3D file with values indexing
                Regions of Interest (ROIs).
            """)
        return message

    def read_inputs_into_2D(self):
        """ Read the inputs and tranform the 4D image into a 2D connectivity
        matrix.
        Returns
        -------
        2D_con_mat : 2D np.array
            2D matrix where rows are seed ROIs and columns the target's voxels

        Notes
        -----
        2D_con_mat is also stored in "temprorary_files" in the results folder
        with the name : subj_2D_connectivity_matrix.npy
        """
        # Name of the 2D connectivity matrix
        self.cmap2D = os.path.join(self.res_dir, self.subj_name + "_cmap2D.npy")
        # Read the brain ribbon mask, which will become the number of columns
        # of the 2D connectivity matrix
        ribbon_data = nib.load(self.in_dict['targetRibbon']).get_data()

        # Create indices for voxels on the brain ribbon
        ind_ribbon = np.where(ribbon_data)
        nvox = np.array(ind_ribbon).shape[1]

        # Now we load the 4D file with the connectivity profiles for each ROI
        co = nib.load(self.in_dict['cmaps4D']).get_data()

        # Record the number of ROIs
        nROIs = co.shape[3]

        # Prepare a zero matrix to store the 2D connectivity matrix
        co_mat = np.zeros((nROIs,nvox))

        # Fill the connectivity matrix
        for i in np.arange(nROIs):
            tmp = co[:,:,:,i]
            co_mat[i,:] = tmp[ind_ribbon]

        # Save the connectivity_matrix in an npy file
        np.save(self.cmap2D, co_mat)

        return co_mat


    def map_ROIs(self):
        ROIs = nib.load(self.in_dict['seedROIs']).get_data()
        # Create an index to the xyz coordinates of the voxels in each ROIs
        ind_xyz_ROIs = np.where(ROIs)
        # Create an index of the ROI associated with each row of the
        # (subsequently) created 2D_connectivity_matrix
        ind_2Drows_to_ROIs_label = ROIs[ind_xyz_ROIs]

        return ind_xyz_ROIs, ind_2Drows_to_ROIs_label


# %%
test1 = tracto_4D("/data/BCBLab/test_COBRA/S1")
mat = test1.read_inputs_into_2D()

mat.shape
print(test1.read_inputs_into_2D.__doc__)
ind, rows = test1.map_ROIs
# %%
st = os.path.join("blabla", "bliblibli")
print("\n" + os.path.dirname(
    os.path.dirname(os.path.join("/data/BCBLab/test_COBRA/S1/", ""))))
# os.rmdir("blibli")
print(st)
tt = """test
test bla"""
type(st)
def func(str):
    assert len(str) > 2, "Error lol"
    print(str)
func("1")
def returns_str():
    return "une string"
assert 2 == 1, returns_str()
