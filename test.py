# -*- coding: utf-8 -*-

import numpy as np
import scipy.stats as ss
import pandas as pd
from scipy.optimize import curve_fit


def rotate_components(phi, gamma = 1.0, q = 50, tol = 1e-6):
    """ Performs rotation of the loadings/eigenvectors
    obtained by means of SVD of the covariance matrix
    of the connectivity profiles.
    https://en.wikipedia.org/wiki/Talk:Varimax_rotation

    Parameters
    ----------
    phi: 2D np.array

    gamma: float
        1.0 for varimax (default), 0.0 for quartimax
    q: int
        number of iterations (default=50)
    tol: float
        tolerance for convergence (default=1e-6)
    """
    p,k = phi.shape
    r = np.eye(k)
    d = 0
    cnt = 0
    for i in np.arange(q):
        cnt = cnt + 1
        d_old = d
        Lambda = np.dot(phi, r)
        u,s,vh = np.linalg.svd(np.dot(
            phi.T,np.asarray(Lambda)**3 - (gamma/p) * np.dot(
                Lambda, np.diag(np.diag(np.dot(Lambda.T,Lambda))))))
        print("Matrix u: ")
        print(u)
        print("Matrix s: ")
        print(s)
        print("Matrix vh: ")
        print(vh)
        r = np.dot(u, vh)
        d = np.sum(s)
        if d_old != 0 and d / d_old < 1 + tol:
            break
    print("Trace rotate_components_START")
    print("Rotation matrix: ")
    print(r)
    print("Loop number: " + str(cnt))
    print("Trace rotate_components_END")
    return np.dot(phi, r)

def fit_power(eigvals_rot):
    """Performs power curve fitting on the rotated eigenvalues
    to obtain the estimated number of PCA components
    Parameters
    ----------
    eigvals_rot: vector
    Returns
    -------
    npc : int
        number of principal components
    """


    L = eigvals_rot

    # Consider only the first 50 eigenvalues, otherwise the
    # curve fitting could be excessively driven by the right
    # tail of the distribution, which has very low values.
    L = L[0:50]

    # Define the fitting function for L
    def powerfunc(x, amp, exponent):
        return amp * (x ** exponent)

    # Define a number of x points corresponding to len(L)
    xL = np.arange(len(L)) + 1

    # Perform curve fitting
    popt, _ = curve_fit(powerfunc, xL, L, method='lm')

    # Calculate the distance from the origin, which is interpreted
    # as the elbow point
    x = np.linspace(1, 50, 1000)
    y = powerfunc(x, *popt)

    d = np.sqrt(x**2 + y**2)
    i0 = np.where(d == np.min(d))

    x0 = x[np.squeeze(i0)]
    # y0 = y[np.squeeze(i0)]

    # Establish the number of principal components on the basis of this
    npc = np.int(np.round(x0))
    return npc


def parcellate_PCA(matrix, mat_type, path_pref, rot='quartimax', eigval_thr=1):
    """ Parellate a 2D similarity matrix with the PCA algorithm
    Parameters
    ----------
    similarity_matrix: 2D np.array
        square similarity_matrix, e.g. correlation matrix
        (It is assumed that the original 2D_connectivity_matrix was
        normalized across ROIs)
    mat_type: str ['covariance', 'correlation']
        the type of similarity matrix. The threshold of eigenvalues will be
        eigval_thr for correlation matrices but
        eigval_thr * mean(eigevalues) for covariance matrices
    rot: str ['quartimax', 'varimax']
        Type of factor rotation
    path_pref: str
        the path and the prefixes to add before the name of files which
        will be created by the function
    eigval_thr: int

    Returns
    -------
    labels: np.array
        Nseed labels (integers) which can be used to assign to each seed
        ROI the value associated to a certain cluster
    """
    if rot == 'quartimax':
        rotation = 0.0
    elif rot == 'varimax':
        rotation = 1.0
    else:
        raise Exception('This factor rotation type is not handled')
    # To have more than just a reference of matrix  in mat
    mat = matrix + 0
    # Get the eigenvalues and eigenvectors of the
    # mat = cov(2D_connectivity_matrix)
    # gamma_eigval, omega_eigvec = np.linalg.eig(mat)
    u, gamma_eigval, omega = np.linalg.svd(mat, full_matrices=True)
    # SVD third output is the transposed of the eigen vectors
    omega_eigvec = omega.T
    if mat_type == "covariance":
        comp_thr = eigval_thr * np.mean(gamma_eigval)
    elif mat_type == "correlation":
        comp_thr = eigval_thr
    else:
        raise Exception('This factor rotation type is not handled')

    # Sort the Gamma_eigval in decreasing order of magnitude, and sort
    # the order of the eigenvectors accordingly
    indsort = np.argsort(gamma_eigval)[::-1]

    # The SSQ_loadings is equal to the eigenvalues of the SM (cov(data))
    # They correspond to the values in the 'Extraction Sum of Squared
    # loadings' in SPSS
    gamma_eigval_sort = gamma_eigval[indsort]
    omega_eigvec_sort = omega_eigvec[:,indsort]

    # We keep only the components which have an eigenvalue above comp_thr
    keep = np.where(gamma_eigval_sort > comp_thr)
    ind = 0
    while gamma_eigval_sort[ind] > comp_thr:
        ind += 1
    gamma_eigval_sort = gamma_eigval_sort[:ind]
    omega_eigvec_sort = omega_eigvec_sort[:,:ind]

    SSQ_loadings = gamma_eigval_sort
    # The matrix of factor laodings (like in SPSS)
    Lambda = omega_eigvec_sort.dot(np.diag(np.sqrt(np.abs(gamma_eigval_sort))))
    print(pd.DataFrame(Lambda))
    # SPSS: The rescaled loadings matrix
    Lambda_rescaled = np.dot(np.sqrt(np.diag(np.diag(cov))), Lambda)

    # SPSS: communalities
    h = [np.sum(gamma_eigval*(omega_eigvec[i]**2)) for i in range(len(omega_eigvec))]

    lambda_rot = rotate_components(Lambda, q = 1000, gamma=rotation)
    print(pd.DataFrame(lambda_rot))
    # Get sum of squared loadings
    SSQ_loadings_rot = np.sum(lambda_rot**2, axis=0)
    print(pd.DataFrame(SSQ_loadings_rot))
    # Sort the SSQ_loadings_rot in descending order to prepare for the
    # power fitting
    SSQ_loadings_rot_sorted = np.sort(SSQ_loadings_rot)
    SSQ_loadings_rot_sorted_descending = SSQ_loadings_rot_sorted[::-1]

    # --------------------------------------------------------------------------
    # (5) Fit a power law to the sorted SSQ_Loadings_rot to Estimate
    #     the number of relevant factors Npc using the fitpower function in
    #     do_PCA_utilities.py (only the first 50 SSQ_Loadings are considered).
    # Returns the number of components to consider: Npc
    # --------------------------------------------------------------------------
    npc = fit_power(SSQ_loadings_rot_sorted_descending)
    print('\n Power fitting of the eigenvalues associated with the rotated')
    print('loadings estimated the presence of ' + str(npc) + ' clusters \n')


    # --------------------------------------------------------------------------
    # (6) Rotate Lambda_Npc = Lambda[:,Npc]
    # Returns the final Factor loadings, defining the clusters
    # --------------------------------------------------------------------------
    lambda_npc = Lambda[:, 0:npc]

    return (lambda_rot, npc)
    # return (lambda_npc, npc)



# path = "/data/neurosynth_data/Wholebrain.csv"
path = "/data/neurosynth_data/Political_test_set.csv"

mat = pd.read_csv(filepath_or_buffer=path, header=0)

pmat = mat[mat.columns[1:]]
pd.DataFrame(pmat)
cov = pmat.cov()
pd.DataFrame(cov)

u, gamma_eigval, omega = np.linalg.svd(cov, full_matrices=True)
# SVD third output is the transposed of the eigen vectors
# omega_eigvec = omega.T
pd.DataFrame(omega.T)
svd_vec = pd.DataFrame(omega.T.dot(np.diag(np.sqrt(np.abs(gamma_eigval)))))
gamma_eigval, omega_eigvec = np.linalg.eig(cov)
eig_vec = pd.DataFrame(omega_eigvec.dot(np.diag(np.sqrt(np.abs(gamma_eigval)))))

np.sign(svd_vec)
[len(np.where(np.sign(svd_vec[[i]]) == -1.0)[0]) for i in range(0, len(eig_vec))]
arr_svd = np.array(svd_vec)
arr_eig = np.array(eig_vec)
arr_csv = np.array(raw_csv)
arr_svd[:,0]
np.subtract([np.sum(np.abs(arr_svd[np.where(np.sign(arr_svd[:,i]) == -1.0)])) for i in range(0, len(arr_svd))], [np.sum(np.abs(arr_svd[np.where(np.sign(arr_svd[:,i]) == 1.0)])) for i in range(0, len(arr_svd))])
np.subtract([np.sum(np.abs(arr_eig[np.where(np.sign(arr_eig[:,i]) == -1.0)])) for i in range(0, len(arr_eig))], [np.sum(np.abs(arr_eig[np.where(np.sign(arr_eig[:,i]) == 1.0)])) for i in range(0, len(arr_eig))])
np.subtract([np.sum(np.abs(arr_csv[np.where(np.sign(arr_csv[:,i]) == -1.0)])) for i in range(0, len(arr_csv))], [np.sum(np.abs(arr_csv[np.where(np.sign(arr_csv[:,i]) == 1.0)])) for i in range(0, len(arr_csv))])

[len(np.where(np.sign(eig_vec[[i]]) == -1.0)[0]) for i in range(0, len(eig_vec))]

raw_csv = pd.read_csv("/home/tolhs/toto.csv", header=None)
[len(np.where(np.sign(raw_csv[[i]]) == -1.0)[0]) for i in range(0, len(eig_vec))]

eigval_thr = 1
if mat_type == "covariance":
    comp_thr = eigval_thr * np.mean(gamma_eigval)
elif mat_type == "correlation":
    comp_thr = eigval_thr
else:
    raise Exception('This factor rotation type is not handled')

# Sort the Gamma_eigval in decreasing order of magnitude, and sort
# the order of the eigenvectors accordingly
indsort = np.argsort(gamma_eigval)[::-1]

# The SSQ_loadings is equal to the eigenvalues of the SM (cov(data))
# They correspond to the values in the 'Extraction Sum of Squared
# loadings' in SPSS
gamma_eigval_sort = gamma_eigval[indsort]
omega_eigvec_sort = omega_eigvec[:,indsort]

# We keep only the components which have an eigenvalue above comp_thr
keep = np.where(gamma_eigval_sort > comp_thr)
ind = 0
while gamma_eigval_sort[ind] > comp_thr:
    ind += 1
gamma_eigval_sort = gamma_eigval_sort[:ind]
omega_eigvec_sort = omega_eigvec_sort[:,:ind]

SSQ_loadings = gamma_eigval_sort
# The matrix of factor laodings (like in SPSS)
Lambda = omega_eigvec_sort.dot(np.diag(np.sqrt(np.abs(gamma_eigval_sort))))

np.linalg.eig(cov)

u, gamma_eigval, omega.T = np.linalg.svd(cov, full_matrices=True)
# SVD third output is the transposed of the eigen vectors
omega_eigvec = omega.T
Lambda = omega_eigvec.dot(np.diag(np.sqrt(np.abs(gamma_eigval))))
h = [np.sum(gamma_eigval*(omega_eigvec[i]**2)) for i in range(len(omega_eigvec))]
Lambda[:,0:2]
Lambda_rescaled = np.dot(np.sqrt(np.diag(np.diag(cov))), Lambda)
pd.DataFrame(Lambda_rescaled)


# norm_Lambda = np.dot(np.sqrt(np.diag(h)), Lambda)
Lambda[0,:]
norm_Lambda = np.array([Lambda[i,:] / np.sqrt(h[i]) for i in range(0, len(h))])
np.dot(np.diag(np.divide(1, np.sqrt(h))), tata)
pd.DataFrame(norm_Lambda)


pd.DataFrame(np.dot(np.diag(np.divide(1, np.sqrt(h))), Lambda))
lambda_rot = rotate_components(np.dot(np.diag(np.divide(1, np.sqrt(h))), tata), q = 1000, gamma=1.0)
pd.DataFrame(lambda_rot)

TRY THIS !!
unorm_rotated = np.dot(lambda_rot, np.sqrt(h)))
pd.DataFrame(unorm_rotated)
(l, npc) = parcellate_PCA(cov, "covariance", "/data/neurosynth_data/res/", "varimax", 1)


SVi = np.sum()

vari_pc1 = [1.830,.916,2.878,1.959,1.831,1.114,2.465,1.851,.371,.708,.527]

vari_pc2 = [1.291,3.566,.564,2.218,1.111,2.776,1.484,2.087,.139,.331,.293]

rot_spss_vari = np.array([vari_pc1, vari_pc2]).T
rot_mat = np.array([[.678, .735], [.735, -.678]])
pd.DataFrame(rot_mat)
rot_norm_lam = np.dot(norm_Lambda[:,0:2], rot_mat)
unorm_rotated_lam = np.dot(rot_norm_lam, np.diag(np.sqrt(h[0:2])))
rot_spss_vari
# This is almost the unrotated components
np.dot(rot_spss_vari, np.linalg.inv(rot_mat))
toto = Lambda[:,0:2]
pd.DataFrame(np.dot(toto, rot_mat))
toto[0,0] * rot_mat[0, 0] + toto[0,1] * rot_mat[1, 0]
pd.DataFrame(Lambda[:,0:2])
np.dot(Lambda[:,0:2], rot_mat)
np.dot(norm_Lambda[:,0:2], rot_mat)

raw_csv = pd.read_csv("/home/tolhs/toto.csv", header=None)
raw_csv[[0,1]]
spss_rot_comp = np.array(raw_csv[[0,1]])
spss_rot_mat = np.array(raw_csv[[2,3]])
spss_rot_mat = np.delete(spss_rot_mat, range(2,11), 0)
inv_rot = np.dot(spss_rot_comp, np.linalg.inv(spss_rot_mat))

tata = Lambda[:,0:2]
tata[:,0] = np.abs(tata[:,0])

np.dot(inv_rot, spss_rot_mat)
np.dot(tata, spss_rot_mat)
nonorm1 = [1.401,3.615,0.739,2.334,1.221,2.839,1.632,2.196,0.161,0.374,0.325]


nonorm2 = [1.747,0.695,2.838,1.819,1.759,0.941,2.37,1.719,0.362,0.686,0.508]

l
l[:,0]
spss = [2.238,2.697,2.727,2.846,2.141,2.437,2.876,2.685,.387,.774,.602]
spss2 = [.098,2.506,-1.080,.808,-.054,1.735,-.083,.756,-.084,-.104,-.038]
ss.pearsonr(np.abs(unorm_rotated_lam[:,0]), np.abs(vari_pc1))
ss.pearsonr(np.abs(unorm_rotated[:,1]), np.abs(vari_pc2))


ss.pearsonr(np.abs(l[:,0]), np.abs(nonorm1))
ss.pearsonr(np.abs(l[:,1]), np.abs(nonorm2))
ss.spearmanr(np.abs(l[:,0]), np.abs(spss))




corr = pmat.corr()
g, omega = np.linalg.eig(cov)

U, S, VT = np.linalg.svd(cov, full_matrices=True)
omega = VT.T
# So g is the vector containing the eigenvalues and SPSS take their absolute val
g = np.abs(S)
# According to SPSS gamma is a diagonal matrix
gamma = np.diag(g)
pd.DataFrame(omega)
# The matrix of factor laodings
lam = np.dot(omega, np.sqrt(gamma))
pd.DataFrame(lam)
rc1 = [1.9947994808957,3.58226658920549,1.83879995484791,2.87432950366585,1.83515385079046,2.97542258530367,2.46069784091457,2.70826051754202,0.29542089941529,0.622287970670245,0.504868309486066]
lam[0]
ss.pearsonr(lam[0], rc1)
ss.spearmanr(np.abs(lam[0]), np.abs(rc1))

jasp_rc1 = [0.327, 0.309, 0.232, 0.866, 0.815, 0.737, 0.850, 0.760, 0.810, 0.828, 0.829]
jasp_rc1_order = [0.810, 0.828, 0.829, 0.327, 0.309, 0.232, 0.866, 0.815, 0.737, 0.850, 0.760]
ss.spearmanr(np.abs(lam[0]), np.abs(jasp_rc1))
ss.spearmanr(np.abs(rc1), np.abs(jasp_rc1))
ss.spearmanr(np.abs(lam[0]), np.abs(jasp_rc1_order))
ss.spearmanr(np.abs(rc1), np.abs(jasp_rc1_order))

jasp_rc2_order = [0.074, -0.040, 0.082, 0.209, 0.346, 0.110, 0.167, 0.221, 0.892, 0.912, 0.897]
ss.pearsonr(np.abs(lam[1]), np.abs(jasp_rc2_order))
jasp_rc2 = [0.209, 0.346, 0.110, 0.167, 0.221, 0.892, 0.912, 0.897, 0.074, -0.040, 0.082]
ss.pearsonr(np.abs(lam[1]), np.abs(jasp_rc2))

pd.DataFrame(lam[:,0:2])
rotation = rotate_components(lam[:,0:2], gamma = 0.0, q=1000)
pd.DataFrame(rotation)
spss_rc1 = [2.238,2.697,2.727,2.846,2.141,2.437,2.876,2.685,0.387,0.774,0.602]
spss_rc2 = [0.098,2.506,-1.08,0.808,-0.054,1.735,-0.083,0.756,-0.084,-0.104,-0.038]

ss.pearsonr(np.abs(rotation[:,1]), np.abs(spss_rc1))
ss.pearsonr(np.abs(rotation[:,0]), np.abs(spss_rc2))


h = [np.sum(g*(omega[i]**2)) for i in range(len(omega))]
np.testing.assert_almost_equal(np.diag(cov), h, decimal=5)
np.isclose(np.diag(cov), h, rtol=1e-07).all()
diag_h = np.diag(h)
norm_lam = np.dot(np.sqrt(diag_h), lam)
rotation = rotate_components(norm_lam, gamma = 0.0)


g, omega = np.linalg.eig(corr)
pd.DataFrame(omega)
U, S, VT = np.linalg.svd(corr, full_matrices=True)
VT.T
rotation = rotatecomponents(g, gamma = 0.0)
