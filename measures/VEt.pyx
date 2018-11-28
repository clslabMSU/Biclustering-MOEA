# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

cimport cython
from libc.math cimport fabs, sqrt
from libc.stdlib cimport malloc, free


def VEt(float[:, :] bicluster):
    """
    Transposed virtual error according to:
    Pontes, B., Giráldez, R., & Aguilar-Ruiz, J. S. (2013). Configurable pattern-based evolutionary biclustering of gene expression data. Algorithms for Molecular Biology, 8(1), 4.

    :param bicluster: bicluster in which the VEt is desired
    :return: VEt of bicluster
    """

    cdef unsigned int I = bicluster.shape[0]
    cdef unsigned int J = bicluster.shape[1]
    cdef float fI = float(I), fJ = float(J)
    cdef float[:, :] bicluster_view = bicluster

    cdef unsigned int i, j = 0

    cdef float VEt_score = 0
    cdef float sum_m1 = 0, sum_m2 = 0
    cdef float mu_rho = 0, sigma_rho = 0

    # I by 1
    cdef float* row_means = <float*>malloc(I*sizeof(float))

    # J by 1
    cdef float* col_means = <float*>malloc(J*sizeof(float))
    cdef float* col_stdevs = <float*>malloc(J*sizeof(float))

    with nogil:

        # calculate column means and standard deviations, calculate row sums
        for j in range(J):
            sum_m1 = 0
            sum_m2 = 0
            for i in range(I):
                if j == 0:
                    row_means[i] = 0
                sum_m1 += bicluster_view[i, j]
                sum_m2 += bicluster_view[i, j]*bicluster_view[i, j]
                row_means[i] += bicluster_view[i, j]
            col_means[j] = sum_m1 / fI
            col_stdevs[j] = sqrt((sum_m2/fI) - (col_means[j]*col_means[j]))

        # finish calculating row means from row sums, calculate sums of first two sample moments
        sum_m1 = 0
        sum_m2 = 0
        for i in range(I):
            row_means[i] /= fI
            sum_m1 += row_means[i]
            sum_m2 += row_means[i]*row_means[i]

        mu_rho = sum_m1 / fI
        sigma_rho = sqrt((sum_m2/fI) - mu_rho*mu_rho)

        # calculate sums in VEt
        for i in range(I):
            for j in range(J):
                VEt_score += fabs(((bicluster_view[i, j] - col_means[j]) / col_stdevs[j]) - ((row_means[i] - mu_rho) / sigma_rho))

        # normalize VEt
        VEt_score /= fI*fJ

        # free memory
        free(row_means)
        free(col_means)
        free(col_stdevs)

    return VEt_score
