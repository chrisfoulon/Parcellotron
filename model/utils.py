# -*- coding: utf-8 -*-

from os.path import splitext, basename

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

def find_in_filename(path, str_arr):
    """ Find string in the filename of path (only in the basename).
        Note that the filename is splitted by underscores.
    Parameters
    ----------
    path : str
        the path of the filename you want to parse
    str_arr : str[]
        the strings you want to find in path
    Returns
    -------
    is_in : bool
        True if the filename contains string
        False if not
    """
    assert len(str_arr) > 0, "The string array is empty"
    assert not "" in str_arr, "str_arr contains an empty string"
    base = basename(path)
    filename = split_ext(base)[0]
    splitted = filename.split('_')
    is_in = True
    for s in str_arr:
        is_in = s in splitted
    return is_in
