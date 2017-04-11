# -*- coding: utf-8 -*-

import abc
import nibabel as nib
import numpy as np
import utils as ut
import os

test_path = "/root/user/filename.ext1.ext2"
os.path.basename(ut.split_ext(test_path)[0])

class parcellotron(metaclass=abc.ABCMeta):
    """ Abstract class of an generic object which will contain informations
    used to parcellate an image.
    """
    @abc.abstractmethod
    def __init__(self, subj_path, modality):
        self.modality = modality
        self.subj_folder = subj_path
        self.subj_name = os.path.basename(subj_name)
        self.root_dir = os.path.dirname(subj_path)
        self.input_dir = os.path.join(subj_path, modality)

    @abc.abstractmethod
    def read_inputs_into_2D(self):
        pass

    @abc.abstractmethod
    def map_ROIs(self):
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
    def __init__(self, subj_path):
        super().__init__(subj_path, self.__class__.__name__)
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
        Choose the hemisphere you want to parcellate (LH for left hemisphere;
        RH for the right one and ALL for both). The function will look for the
        file containing 'LH', 'RH' or none of those (for ALL)
        Returns
        -------

        """
        print("Olol")

    def map_ROIs(self):
        print("ok")


test = tracto_4D("test", "test_dir")
test.map_ROIs()
import os
st = os.path.join("blabla", "bliblibli")
type(st)
def func(str):
    assert len(str) > 2, "Error lol"
    print(str)
func("eéé")
