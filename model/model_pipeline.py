# -*- coding: utf-8 -*-

import argparse
import parcellotron as pa
import matrix_tranformations as mt
import similarity_matrices as sm
import parcellation_methods as pm

parser = argparse.ArgumentParser(description="Calculate the parcellation of\
                                 brain images")

# The available modalities
modality_arr = ['tracto_4D', 'tracto_mat']
sim_mat_arr = ['distance', 'covariance', 'correlation']
mat_transform_arr = ['log2', 'zscore', 'log2_zscore', 'none']
parcellation_method_arr = ['KMeans', 'PCA']

parser.add_argument("subj_path", type=str, help="the subject folder path")
parser.add_argument("modality", type=str, help="the input modality",
                    choices=modality_arr)
parser.add_argument("similarity_matrix", type=str, help="the input modality",
                    choices=modality_arr, default='correlation')
parser.add_argument("matrix_transformation", type=str,
                    help="the transformation(s) to apply to the similarity \
                    matrix", choices=mat_transform_arr, default='log2_zscore')
parser.add_argument("parcellation_method", type=str,
                    help="the parcellation methods to use",
                    choices=parcellation_method_arr)

# group = parser.add_mutually_exclusive_group()
# group.add_argument("-v", "--verbose", action="store_true")
# group.add_argument("-q", "--quiet", action="store_true")
# parser.add_argument("y", type=int, help="the exponent")
# answer = args.x**args.y
args = parser.parse_args()

def parcellate(path, mod, transformation, sim_mat, method):
    # modality choice
    if args.modality == 'tracto_4D':
        subj_obj = pa.tracto_4D(mod)
        mat_2D = subj_obj.co_mat_2D
    else:
        raise Exception("bad modality")

    # matrix transformation
    if transformation in ['log2', 'log2_zscore']:
        mat_2D = mt.matrix_log2(mat_2D)
    if transformation in ['zscore', 'log2_zscore']:
        mat_2D = mt.matrix_zscore(mat_2D)

    # similarity_matrix
    if sim_mat == 'covariance':
        sim = sm.similarity_covariance(mat_2D)
    if sim_mat == 'correlation':
        sim = sm.similarity_correlation(mat_2D)
    if sim_mat == 'distance':
        sim = sm.similarity_distance(mat_2D)

    # parcellation
    if method == 'KMeans':
        labels = pm.parcellate_KMeans(sim, 10)
    else:
        raise Exception("Not yet implemented")

    return labels

import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import time

t0 = time.time()
subj_labels = parcellate(args.subj_path, args.modality,
                     args.matrix_transformation,
                     args.similarity_matrix, args.parcellation_method)
t1 = time.time()

print("KMeans performed in %.3f s" % (t1 - t0))

IDX_CLU = np.argsort(labels)

similarity_matrix_reordered = similarity_matrix[IDX_CLU,:][:,IDX_CLU]

plt.imshow(similarity_matrix_reordered, interpolation='none')
plt.show()
