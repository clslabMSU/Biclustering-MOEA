import numpy as np
from os import name
from scipy.stats import pearsonr, rankdata
from typing import List, Union


def build_function(function_name: str, build_file: str, build_dir: str = "/measures"):
    print("Building %s function... This should only happen once." % function_name)
    from os import getcwd, system
    from shutil import rmtree
    from sys import executable
    build_path = getcwd().replace(" ", "\\ ") + build_dir
    if name == "nt":
        print("Running on Windows...") # unfortunately
        command = "\"{python}\" \"{path}\\{file}\" build_ext --inplace > NUL".format(python=executable, path=build_path.replace("\\ ", " ").replace("/", "\\"), file=build_file)
    else:
        command = "{python} {path}/{file} build_ext --inplace >/dev/null".format(python=executable, path=build_path, file=build_file)
    print("Executing command: " + command)
    system(command)
    if name != "nt":
        print("Removing build files...")
        rmtree("build")
    print("Finished build.")


try:
    from measures.ASR import ASR as ASR_cython
except ImportError:
    build_function("ASR", "build_ASR.py")
finally:
    from measures.ASR import ASR as ASR_cython

try:
    from measures.SCS import SCS as SCS_cython
except ImportError:
    build_function("SCS", "build_SCS.py")
finally:
    from measures.SCS import SCS as SCS_cython

try:
    from measures.VEt import VEt as VEt_cython
except ImportError:
    build_function("VEt", "build_VEt.py")
finally:
    from measures.VEt import VEt as VEt_cython

try:
    from measures.TPC import TPC as TPC_cython
except ImportError:
    build_function("TPC", "build_TPC.py")
finally:
    from measures.TPC import TPC as TPC_cython

try:
    from measures.GeneVariance import GeneVariance as GeneVariance_cython
except ImportError:
    build_function("GeneVariance", "build_GeneVariance.py")
finally:
    from measures.GeneVariance import GeneVariance as GeneVariance_cython

###############################################################################################################################################
############################################################# External Validation #############################################################
###############################################################################################################################################


def MatchScore(B1: list, B2: list) -> float:
    ms = 0.0
    for i in B1:
        ms += max([float(len(list(set(i["genes"]).intersection(set(j["genes"]))))) /
                   float(len(list(set(i["genes"]).union(set(j["genes"])))))
                   for j in B2])
    return ms / len(B1)


def Recovery(discovered: list, expected: list) -> float:
    return MatchScore(expected, discovered)


def Relevance(discovered: list, expected: list) -> float:
    return MatchScore(discovered, expected)


###############################################################################################################################################
############################################################# Internal Validation #############################################################
###############################################################################################################################################


def MSR(bicluster: np.ndarray) -> float:
    """
    Mean Squared Residue Score
    Cheng, Y., & Church, G. M. (2000, August). Biclustering of expression data. In Ismb (Vol. 8, No. 2000, pp. 93-103).

    :param bicluster: - np.ndarray of expression levels of bicluster
    :return: - mean squared residue score, lower is better
    """
    column_means = np.mean(bicluster, axis=0)
    row_means = np.mean(bicluster, axis=1)
    bicluster_mean = np.mean(bicluster.flatten())
    msr = 0
    for i in range(bicluster.shape[0]):
        for j in range(bicluster.shape[1]):
            msr += (bicluster[i, j] - row_means[i] - column_means[j] + bicluster_mean)**2
    return msr / (bicluster.shape[0] * bicluster.shape[1])


def SMSR(bicluster: np.ndarray) -> float:
    """
    Scaled Mean Squared Residue Score
    Mukhopadhyay, A., Maulik, U., & Bandyopadhyay, S. (2009). A novel coherence measure for discovering scaling biclusters from gene expression data. Journal of Bioinformatics and Computational Biology, 7(05), 853-868.

    :param bicluster: - np.ndarray of expression levels of bicluster
    :return: - scaled mean squared residue score, lower is better
    """
    column_means = np.mean(bicluster, axis=0)
    row_means = np.mean(bicluster, axis=1)
    bicluster_mean = np.mean(bicluster.flatten())
    smsr = 0
    for i in range(bicluster.shape[0]):
        for j in range(bicluster.shape[1]):
            smsr += (row_means[i] * column_means[j] - bicluster[i, j] * bicluster_mean)**2 / (row_means[i]**2 * column_means[j]**2)
    return smsr / (bicluster.shape[0] * bicluster.shape[1])


def VE(bicluster: np.ndarray) -> float:
    """
    Virtual Error of a bicluster
    Divina, F., Pontes, B., Giráldez, R., & Aguilar-Ruiz, J. S. (2012). An effective measure for assessing the quality of biclusters. Computers in biology and medicine, 42(2), 245-256.

    :param bicluster: - np.ndarray of expression levels of bicluster
    :return: virtual error score, lower is better
    """
    if bicluster.shape[0] <= 1 or bicluster.shape[1] <= 1:
        return np.inf
    rho = np.mean(bicluster, axis=0)
    rho_std = np.std(rho)
    if rho_std != 0:
        rho_hat = (rho - np.mean(rho)) / np.std(rho)
    else:
        rho_hat = (rho - np.mean(rho))
    bic_hat = _standardize_bicluster_(bicluster)
    ve = 0
    for i in range(bicluster.shape[0]):
        for j in range(bicluster.shape[1]):
            ve += abs(bic_hat[i, j] - rho_hat[j])
    ve /= (bicluster.shape[0] * bicluster.shape[1])
    return ve


def VEt(bicluster: np.ndarray) -> float:
    """
    Transposed virtual error of a bicluster
    Pontes, B., Giráldez, R., & Aguilar-Ruiz, J. S. (2010, September). Measuring the Quality of Shifting and Scaling Patterns in Biclusters. In PRIB (pp. 242-252). Chicago


    :param bicluster: - np.ndarray of expression levels of bicluster
    :return: transposed virtual error, lower is better
    """
    return VE(np.transpose(bicluster))


def VEt_fast(bicluster: np.ndarray) -> float:
    return VEt_cython(bicluster)


def ASR_fast(bicluster: np.ndarray) -> float:
    return ASR_cython(bicluster)


#def ASR(bicluster: np.ndarray) -> float:
#    """
#    Average Spearman's Rho for a bicluster.
#    Ayadi, W., Elloumi, M., & Hao, J. K. (2009). A biclustering algorithm based on a Bicluster Enumeration Tree: application to DNA microarray data. BioData mining, 2(1), 9.
#
#    :param bicluster: - np.ndarray of expression levels of bicluster
#    :return: average spearman's rho, close to 1 or -1 is better
#    """
#    if bicluster.shape[0] <= 1 or bicluster.shape[1] <= 1:
#        return 0
#    spearman_genes = 0
#    spearman_samples = 0
#    row_ranks = np.array([rankdata(bicluster[i, :]) for i in range(bicluster.shape[0])])
#    col_ranks = np.array([rankdata(bicluster[:, j]) for j in range(bicluster.shape[1])])
#    for i in range(bicluster.shape[0] - 1):
#        for j in range(i+1, bicluster.shape[0]):
#            spearman_genes += spearman(bicluster[i, :], bicluster[j, :], row_ranks[i, :], row_ranks[j, :])
#    for k in range(bicluster.shape[1] - 1):
#        for l in range(k+1, bicluster.shape[1]):
#            spearman_samples += spearman(bicluster[:, k], bicluster[:, l], col_ranks[k, :], col_ranks[l, :])
#    spearman_genes /= bicluster.shape[0]*(bicluster.shape[0]-1)
#    spearman_samples /= bicluster.shape[1]*(bicluster.shape[1]-1)
#    asr = 2*max(spearman_genes, spearman_samples)
#    return asr
#
#
#def spearman(x: np.ndarray, y: np.ndarray, rx: Union[np.ndarray, None] = None, ry: Union[np.ndarray, None] = None) -> float:
#    """
#    Spearman's Rho for two vectors.
#    Ayadi, W., Elloumi, M., & Hao, J. K. (2009). A biclustering algorithm based on a Bicluster Enumeration Tree: application to DNA microarray data. BioData mining, 2(1), 9.
#
#    :param x: first vector
#    :param y: second vector
#    :param rx: ranks of first vector, optional
#    :param ry: ranks of second vector, optional
#    :return: spearman's rho between vectors, close to 1 or -1 is better
#    """
#    assert(x.shape == y.shape)
#    if rx is None: rx = rankdata(x)
#    if ry is None: ry = rankdata(y)
#    m = len(x)
#    coef = 6.0/(m**3-m)
#    ans = 0
#    for k in range(m):
#        ans += (rx[k] - ry[k])**2
#    return 1 - coef*ans


def _standardize_bicluster_(bicluster: np.ndarray) -> np.ndarray:
    """
    Standardize a bicluster by subtracting the mean and dividing by standard deviation.
    Pontes, B., Girldez, R., & Aguilar-Ruiz, J. S. (2015). Quality measures for gene expression biclusters. PloS one, 10(3), e0115497.

    Note that UniBic synthetic data was generated with mean 0 and standard deviation 1, so it is already standardized.

    :param bicluster: np.ndarray of expression levels of bicluster
    :return: standardized bicluster
    """
    bic = np.copy(bicluster)
    for i in range(bic.shape[0]):
        gene = bic[i, :]
        std = np.std(gene)
        if std != 0:
            bic[i, :] = (gene - np.mean(gene)) / std
        else:
            bic[i, :] = (gene - np.mean(gene))
    return bic


def bicluster_volume(bicluster: np.ndarray, weights: Union[List[float], None] = None) -> float:
    """
    Calculate bicluster volume based on:
    Pontes, B., Giráldez, R., & Aguilar-Ruiz, J. S. (2013). Configurable pattern-based evolutionary biclustering of gene expression data. Algorithms for Molecular Biology, 8(1), 4.

    :param bicluster: np.ndarray of expression levels of bicluster
    :param weights: user-defined weight vector, or None (volume will return -2 if weights are None)
    :return:
    """
    if weights is None:
        weights = np.zeros(len(bicluster.shape))
    return sum([-np.log(bicluster.shape[i])/(np.log(bicluster.shape[i]) + weights[i]) for i in range(len(bicluster.shape))])


def SCS_fast(bicluster: np.ndarray) -> float:
    return SCS_cython(bicluster)


# def submatrix_correlation_score(bicluster: np.ndarray) -> float:
#     """
#     Submatrix correlation score
#
#     Yang, W. H., Dai, D. Q., & Yan, H. (2011). Finding correlated biclusters from
#     gene expression data. IEEE Transactions on Knowledge and Data Engineering, 23(4), 568-584.
#
#     :param bicluster:
#     :return:
#     """
#     row_wise_cors = np.zeros((bicluster.shape[0], bicluster.shape[0]))
#     col_wise_cors = np.zeros((bicluster.shape[1], bicluster.shape[1]))
#
#     for i in range(bicluster.shape[0] - 1):
#         row = bicluster[i, :]
#         for j in range(i + 1, bicluster.shape[0]):
#             cor = abs(pearsonr(row, bicluster[j, :])[0])
#             row_wise_cors[i, j] = cor
#             row_wise_cors[j, i] = cor
#
#     for i in range(bicluster.shape[1] - 1):
#         col = bicluster[:, i]
#         for j in range(i + 1, bicluster.shape[1]):
#             cor = abs(pearsonr(col, bicluster[:, j])[0])
#             col_wise_cors[i, j] = cor
#             col_wise_cors[j, i] = cor
#
#     SJ = [1 - np.sum(row_wise_cors[:, i])/(bicluster.shape[0]-1) for i in range(bicluster.shape[0])]
#     SI = [1 - np.sum(col_wise_cors[:, i])/(bicluster.shape[1]-1) for i in range(bicluster.shape[1])]
#
#     S_row = min(SJ)
#     S_col = min(SI)
#
#     return min(S_row, S_col)

def TPC_fast(bicluster: np.ndarray) -> float:
    return TPC_cython(bicluster)


def GeneVariance_fast(bicluster: np.ndarray) -> float:
    return GeneVariance_cython(bicluster)
