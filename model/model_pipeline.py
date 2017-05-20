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
                    help="A prefix to find a particular seedROIs file")
parser.add_argument("-tp", "--target_pref", type=str,
                    help="A prefix to find a particular targetMask file")
parser.add_argument("modality", type=str, help="the input modality",
                    choices=modality_arr)
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

def memory_usage():
    # return the memory usage in MB
    import psutil
    process = psutil.Process(os.getpid())
    mem = process.memory_full_info()[0] / float(2 ** 20)
    return mem




def parcellate_obj(files_arr, mod, transformation, sim_mat, method,
                   param_parcellate, seed_pref='', target_pref=''):

    for dir in files_arr:
        if mod == 'Tracto_4D':
            subj_obj = pa.Tracto_4D(dir, group_level=False,
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


# def parcellate_subj(path, mod, transformation, sim_mat, method,
#                         seed_pref='',
#                         target_pref=''):
#     """ Do the parcellation on one subject according the options given in
#     parameters
#
#     Parameters
#     ----------
#     path: str
#         The path to the folder containing the different modality folders which
#         contain the inputs.
#     mod: str
#         the name of the modality
#     transformation: str {'log2', 'zscore', 'log2_zscore', 'none'}
#         the transformation to do to the 2D connectivity matrix
#     sim_mat: str {'distance', 'covariance', 'correlation'}
#         the type of similarity matrix you want
#     method: str {'KMeans', 'PCA'}
#         the parcellation method
#     """
#     import psutil
#     print(memory_usage())
#     mem = psutil.virtual_memory()
#     print(mem)
#     t0 = time.time()
#     # modality choice
#     if mod == 'Tracto_4D':
#         subj_obj = pa.Tracto_4D(path, group_level=False,
#                                 seed_pref=seed_pref,
#                                 target_pref=target_pref)
#         mat_2D = subj_obj.co_mat_2D
#     else:
#         raise Exception(mod + " is not yet implemented")
#
#
#
#     # matrix transformation
#     if transformation in ['log2', 'log2_zscore']:
#         mat_2D = mt.matrix_log2(mat_2D)
#     if transformation in ['zscore', 'log2_zscore']:
#         mat_2D = mt.matrix_zscore(mat_2D)
#
#     # similarity_matrix
#     if sim_mat == 'covariance':
#         sim = sm.similarity_covariance(mat_2D)
#     if sim_mat == 'correlation':
#         sim = sm.similarity_correlation(mat_2D)
#     if sim_mat == 'distance':
#         sim = sm.similarity_distance(mat_2D)
#
#     # parcellation
#     if method == 'KMeans':
#         labels = pm.parcellate_KMeans(sim, 10)
#     elif method == 'PCA':
#         labels = pm.parcellate_PCA(sim)
#     else:
#         raise Exception(method + " is not yet implemented")
#
#     t1 = time.time()
#     print("KMeans performed in %.3f s" % (t1 - t0))
#
#     IDX_CLU = np.argsort(labels)
#
#     similarity_matrix_reordered = sim[IDX_CLU,:][:,IDX_CLU]
#
#     plt.imshow(similarity_matrix_reordered, interpolation='none')
#     plt.show()
#     print(memory_usage())
#     mem = psutil.virtual_memory()
#     print(mem)
#
#     return labels
#
# def parcellate_group(path, mod, transformation, sim_mat, method,
#                         seed_pref='',
#                         target_pref=''):
#     """ Do the parcellation on the group level according the options given in
#     parameters. The software will loop on all the folders of the root
#
#     Parameters
#     ----------
#     path: str
#         The path to the folder containing the different subject folders
#     mod: str
#         the name of the modality
#     transformation: str {'log2', 'zscore', 'log2_zscore', 'none'}
#         the transformation to do to the 2D connectivity matrix
#     sim_mat: str {'distance', 'covariance', 'correlation'}
#         the type of similarity matrix you want
#     method: str {'KMeans', 'PCA'}
#         the parcellation method
#     """
#     labels_dict = {}
#     print(sorted(os.listdir(path)))
#     for subj in sorted(os.listdir(path)):
#         if subj == "_group_level":
#             continue
#         subject_path = os.path.join(path, subj)
#         t0 = time.time()
#         # modality choice
#         if mod == 'Tracto_4D':
#             subj_obj = pa.Tracto_4D(subject_path, group_level=True,
#                                     seed_pref=seed_pref,
#                                     target_pref=target_pref)
#             mat_2D = subj_obj.co_mat_2D
#         else:
#             raise Exception(mod + " is not yet implemented")
#
#         print(subj_obj.__doc__)
#
#         # matrix transformation
#         if transformation in ['log2', 'log2_zscore']:
#             mat_2D = mt.matrix_log2(mat_2D)
#         if transformation in ['zscore', 'log2_zscore']:
#             mat_2D = mt.matrix_zscore(mat_2D)
#
#         # similarity_matrix
#         if sim_mat == 'covariance':
#             sim = sm.similarity_covariance(mat_2D)
#         if sim_mat == 'correlation':
#             sim = sm.similarity_correlation(mat_2D)
#         if sim_mat == 'distance':
#             sim = sm.similarity_distance(mat_2D)
#
#         # parcellation
#         if method == 'KMeans':
#             labels = pm.parcellate_KMeans(sim, 10)
#         elif method == 'PCA':
#             labels = pm.parcellate_PCA(sim)
#         else:
#             raise Exception(method + " is not yet implemented")
#
#         t1 = time.time()
#         print("KMeans performed in %.3f s" % (t1 - t0))
#
#         IDX_CLU = np.argsort(labels)
#
#         similarity_matrix_reordered = sim[IDX_CLU,:][:,IDX_CLU]
#
#         plt.imshow(similarity_matrix_reordered, interpolation='none')
#         plt.show()
#
#         labels_dict[subj_obj.subj_name] = labels
#         print(memory_usage())
#
#     return labels_dict

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
    return_arr = [files_arr, mod, tr, sim, meth, seed, tar]
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
