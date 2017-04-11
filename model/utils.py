# -*- coding: utf-8 -*-

from os.path import splitext

def split_ext(path):
    """ Split the file extension from the path even in case of double extension.
    (like .nii.gz)
    Parameters
    ----------
    path : Full path or just file name
    Returns
    -------
    couple of file name(or path) and the extension(s)
    """
    if len(path.split('.')) > 2:
        return path.split('.')[0],'.'.join(path.split('.')[-2:])
    return splitext(path)
