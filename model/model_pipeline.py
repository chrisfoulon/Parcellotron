# -*- coding: utf-8 -*-

import argparse

parser = argparse.ArgumentParser(description="Calculate the parcellation of\
                                 brain images")

# The available modalities
modality_arr = ['tracto_4D', 'tracto_matrix']
sim_mat_arr = ['distance', 'covariance', 'correlation']
mat_transform_arr = ['log2', 'zscore', 'log2_zscore']
parcellation_methods_arr = ['KMeans', 'PCA']

parser.add_argument("modality", type=str, help="the input modality",
                    choices=modality_arr)
parser.add_argument("similarity_matrix", type=str, help="the input modality",
                    choices=modality_arr, default='correlation')
parser.add_argument("matrix_transformation", type=str,
                    help="the transformation(s) to apply to the similarity \
                    matrix", choices=mat_transform_arr, default='log2_zscore')
parser.add_argument("parcellation_methods", type=str,
                    help="the parcellation_methods to use",
                    choices=parcellation_methods_arr)

group = parser.add_mutually_exclusive_group()
group.add_argument("-v", "--verbose", action="store_true")
group.add_argument("-q", "--quiet", action="store_true")
parser.add_argument("y", type=int, help="the exponent")
args = parser.parse_args()
answer = args.x**args.y

if args.quiet:
    print(answer)
elif args.verbose:
    print("{} to the power {} equals {}".format(args.x, args.y, answer))
else:
    print("{}^{} == {}".format(args.x, args.y, answer))
