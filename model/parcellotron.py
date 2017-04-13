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
    """
    @abc.abstractmethod
    def __init__(self, subj_path, modality):
        self.modality = modality
        self.subj_folder = subj_path
        self.subj_name = os.path.basename(subj_path)
        self.root_dir = os.path.dirname(subj_path)
        self.input_dir = os.path.join(subj_path, modality)
        print(self.input_dir)
        self.res_dir = os.path.join(self.input_dir, software_name + "_results")
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
    """
    def __init__(self, subj_path, hemi='ALL'):
        super().__init__(subj_path, self.__class__.__name__)
        assert hemi == 'LH' or hemi == 'RH' or hemi == 'ALL', "The hemi \
        parameter can be LH RH or ALL"
        self.hemi = hemi
        print(self.subj_name)

    def read_inputs_into_2D(self, subj_path, hemi):
        """ Read the inputs and tranform the 4D image into a 2D connectivity
        matrix.
        Parameters
        ----------
        subj_path : str
            The path to the subject folder, containing the 4D image.
            (The filename must contain "4Dcmaps" separated by underscores)
        hemi : {'LH', 'RH', 'ALL'}, optional (default 'ALL')
            Choose the hemisphere you want to parcellate (LH for left
            hemisphere; RH for the right one and ALL for both). The function
            will look for the file containing 'LH', 'RH' or none of those
            (for 'ALL')
        Returns
        -------
        2D_con_mat : 2D np.array
            2D matrix where rows are seed ROIs and columns the target's voxels

        Notes
        -----
        2D_con_mat is also stored in "temprorary_files" in the results folder
        with the name : subj[_hemi]_2D_connectivity_matrix.npy
        """
        print("Olol")

    def map_ROIs(self):
        print("ok")

    def verify_input_folder(self, in_path, hemi='ALL'):
        boo = False

        if hemi == 'ALL':

        else:

        return(False)

    def inputs_needed(self):
        """
        Returns
        -------
        message : str
            A string describing the inputs you need for this modality
        """
        message = textwrap.dedent("""\
            Inputs needed for this modality :
            1) subj[_hemi]_4Dcmaps.nii[.gz] a 4D image with a connectivity map
                for each time point
            2) subj[_hemi]_ribbon.nii[.gz] the 3D binary mask of the brain ribbon
            3) subj[_hemi]_ROIs.nii[.gz] 3D file with values indexing
                Regions of Interest (ROIs).
            [_hemi] can be LH or RH if you want to calculate the parcellation
            for the left or the right hemisphere respectively""")
        return message

# %%
test = tracto_4D("/data/BCBLab/test_COBRA/S1").inputs_needed()
test.map_ROIs()
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
import re
st1 = "subj_LH_cmaps4D.nii.gz"
st2 = "subj_cmaps4D_LH.nii.gz"

re.search(r"[(_LH)(_cmaps4D)]", st1)
