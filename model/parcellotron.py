# -*- coding: utf-8 -*-

import abc
import nibabel as nib
import numpy as np
import utils as ut
import os
import glob
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
        self.root_dir = os.path.dirname(subj_path)
        self.input_dir = os.path.join(subj_path, modality)
        self.res_dir = os.path.join(self.input_dir, software_name + "_results")
        self.in_dict = {}
        if not os.path.exists(self.res_dir):
            os.mkdir(self.res_dir)
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
            tmp_boo = ut.find_in_filename(in_path, k)
            if not tmp_boo:
                print('I did not find the ' + k + ' file.')
            boo = boo and tmp_boo
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

    def read_inputs_into_2D(self, subj_path):
        """ Read the inputs and tranform the 4D image into a 2D connectivity
        matrix.
        Parameters
        ----------
        subj_path : str
            The path to the subject folder, containing the 4D image.
            (The filename must contain "4Dcmaps" separated by underscores)
        Returns
        -------
        2D_con_mat : 2D np.array
            2D matrix where rows are seed ROIs and columns the target's voxels

        Notes
        -----
        2D_con_mat is also stored in "temprorary_files" in the results folder
        with the name : subj_2D_connectivity_matrix.npy
        """
        print("Olol")

    def map_ROIs(self):
        print("ok")

# %%
test1 = tracto_4D("/data/BCBLab/test_COBRA/S1")

# %%
import os
st = os.path.join("blabla", "bliblibli")

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
