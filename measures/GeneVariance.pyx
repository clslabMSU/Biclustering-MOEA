# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

cimport cython
from libc.stdlib cimport malloc, free


def GeneVariance(float[:, :] bicluster):
    """
    Gene variance according to:
    Pontes, B., Giráldez, R., & Aguilar-Ruiz, J. S. (2013). Configurable pattern-based evolutionary biclustering of gene expression data. Algorithms for Molecular Biology, 8(1), 4.

    :param bicluster: bicluster in which the gene variance is desired
    :return: gene variance of bicluster
    """

    cdef unsigned int I = bicluster.shape[0]
    cdef unsigned int J = bicluster.shape[1]
    cdef float fI = float(I), fJ = float(J)
    cdef float[:, :] bicluster_view = bicluster

    cdef unsigned int i, j = 0
    cdef float total = 0

    # I by 1
    cdef float* row_means = <float*>malloc(I*sizeof(float))

    with nogil:

        # calculate row means
        for i in range(I):
            for j in range(J):
                total += bicluster_view[i, j]
            row_means[i] = total / fJ
            total = 0

        for i in range(I):
            for j in range(J):
                total += (bicluster_view[i, j] - row_means[i])*(bicluster_view[i, j] - row_means[i])
        total /= (fI*fJ)

        free(row_means)

    return total

