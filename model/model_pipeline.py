# -*- coding: utf-8 -*-

import argparse
import numpy as np
import time
import os

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

import parcellobject as pa
import matrix_transformations as mt
import similarity_matrices as sm
import parcellation_methods as pm

parser = argparse.ArgumentParser(description="Calculate the parcellation of\
                                 brain images")
def temp_visualization(path):
    mat = np.load(path)
    plt.imshow(mat, aspect='auto', interpolation='none');
    plt.show()
# The available modalities
modality_arr = ['Tracto_4D', 'Tracto_mat']
sim_mat_arr = ['distance', 'covariance', 'correlation']
mat_transform_arr = ['log2', 'zscore', 'log2_zscore', 'none']
parcellation_method_arr = ['KMeans', 'PCA']
rotation_arr = ['quartimax', 'varimax']

# I need subcommands
# https://docs.python.org/3/library/argparse.html#module-argparse

group = parser.add_mutually_exclusive_group()
group.add_argument('-s', '--subject', type=str, help="the subject folder path")
group.add_argument('-g', '--group', type=str,
                   help="the root folder path, containing the subject folders")
# parser.add_argument("subj_path", type=str, help="the subject folder path")
parser.add_argument("-sp", "--seed_pref", type=str,
                    help="A prefix to find a particular seedROIs/seedMask file")
parser.add_argument("-tp", "--target_pref", type=str,
                    help="A prefix to find a particular targetMask file")
# parser.add_argument("modality", type=str, help="the input modality",
#                     choices=modality_arr)

parser_4D = sub_parsers.add_parser('-Tracto_4D',
                                   help='Tractography in 4D format')
parser_mat = sub_parsers.add_parser('-Tracto_mat',
                                    help='Tractography in probtrackx matrix format')
parser_mat.add_argument('ROIs_size', type=int,
                       help='The size in cubic millimeters')

parser.add_argument("similarity_matrix", type=str,
                    help="type of similarity_matrix you want",
                    choices=sim_mat_arr)
parser.add_argument("-t", "--transform", type=str,
                    help="the transformation(s) to apply to the similarity \
                    matrix", choices=mat_transform_arr)
sub_parsers = parser.add_subparsers(help='Choose the parcellation method',
                                    dest='parcellation_method')

parser_PCA = sub_parsers.add_parser('PCA', help='Parcellate your data using \
                                    the PCA algorithm')
parser_PCA.add_argument('-r', '--rotation', help='Select the factor rotation',
                        type=str, default='quartimax', choices=rotation_arr)
parser_KMeans = sub_parsers.add_parser('KMeans', help="Parcellate your data \
                                       using the KMeans algorithm")
parser_KMeans.add_argument('num_clu', help='Choose the number of cluster you \
                           want to find with the KMeans', type=int)
# parser.add_argument("parcellation_method", type=str,
#                     help="the parcellation methods to use",
#                     choices=parcellation_method_arr)

# group = parser.add_mutually_exclusive_group()
# group.add_argument("-v", "--verbose", action="store_true")
# group.add_argument("-q", "--quiet", action="store_true")
# parser.add_argument("y", type=int, help="the exponent")
# answer = args.x**args.y
args = parser.parse_args()

# Mettre la méthode the parcellisation avant la séléction de la transformation et de
# la similarity. Distance matrix il faut le disable

def parcellate_obj(files_arr, mod, size, transformation, sim_mat, method,
                   param_parcellate, seed_pref='', target_pref=''):

    for dir in files_arr:
        if mod == 'Tracto_4D':
            subj_obj = pa.Tracto_4D(dir, size,
                                    group_level=False,
                                    seed_pref=seed_pref,
                                    target_pref=target_pref)
        elif mod == "Tracto_mat":
            subj_obj = pa.Tracto_mat(dir, group_level=False,
                                    seed_pref=seed_pref,
                                    target_pref=target_pref)
        else:
            raise Exception(mod + " is not yet implemented")


        mat_2D = subj_obj.co_mat_2D

        subj_obj.mat_transform(transformation, mat_2D)
        subj_obj.similarity(sim_mat, subj_obj.tr_mat_2D)
        labels = subj_obj.parcellate(method, subj_obj.sim_mat, param_parcellate)

        for el in subj_obj.temp_dict.values():
            if os.path.exists(el):
                print(el)
                temp_visualization(el)
We should create a third option 'loop' to just calculate the parcellation of every
subjects of the root folder but don't perform any group level analysis on them. 
# We launch the right function on the parameters
print(args)

def filter_args(args):
    path = args.subject
    group = args.group

    if group != None:
        files_arr = [dir for dir in sorted(
            os.listdir(path)) if dir != '_group_level']
    else:
        files_arr = [path]
    roisize = args.ROIs_size
    mod = args.modality
    tr = args.transform
    sim = args.similarity_matrix
    meth = args.parcellation_method
    if 'rotation' in args:
        param_parcellate = args.rotation
    else:
        param_parcellate = args.num_clu
    seed = args.seed_pref
    tar = args.target_pref

    if tr == None:
        if meth == 'PCA':
            tr = 'log2_zscore'
        else:
            tr = 'log2'
    if sim == None:
        if meth == 'PCA':
            sim = 'covariance'
        else:
            sim = 'correlation'
            print("Ok it works")
    return_arr = [files_arr, mod, roisize, tr, sim, meth, seed, tar]
    return return_arr

# The parcellation method will determine the default values of transformations
# and similarity_matrix
# subj_labels = parcellate_obj(args.subject,
#                              args.group,
#                              args.modality,
#                              args.transform,
#                              args.similarity_matrix,
#                              args.parcellation_method,
#                              param_parcellate,
#                              args.seed_pref,
#                              args.target_pref)

subj_labels = parcellate_obj(*filter_args(args))
