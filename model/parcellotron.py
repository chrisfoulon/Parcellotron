# -*- coding: utf-8 -*-

import abc
import nibabel as nib
import numpy as np

class parcellotron(metaclass=abc.ABCMeta):
    """ Abstract class of an generic object which will contain informations
    used to parcellate an image.
    """
    @abc.abstractmethod
    def __init__(self, subj_path):
        self.subj_folder = subj_path
        self.subj_name = name
        self.dir = dir_path

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
    The
    """
    def __init__(self, name):
        super().__init__(subj_path)
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
