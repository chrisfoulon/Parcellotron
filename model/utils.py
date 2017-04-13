# -*- coding: utf-8 -*-

from os.path import splitext, basename
import glob

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

def find_in_filename(path, string):
    """ Find string in the filename of path (only in the basename).
        Note that the filename is splitted by underscores.
    Parameters
    ----------
    path : str
        the path of the filename you want to parse
    string : str
        the string you want to find in path
    Returns
    -------
    file_path : str
        the path to the file if ONE filename contains string
        empty str if not
        Throw an error if several files contain string
    """
    # join is here to handle folder paths with or without '/' at the end
    arr = glob.glob(os.path.join(p, "") + '*' + string + '*')
    if len(arr) == 0:
        return ""
    elif len(arr) == 1:
        return arr[0]
    else:
        raise Exception("I found several files corresponding to this pattern")
