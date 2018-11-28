# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

cimport cython
from libc.math cimport fabs, sqrt


def SCS(float[:, :] bicluster):
    """
    Submatrix correlation score

    Yang, W. H., Dai, D. Q., & Yan, H. (2011). Finding correlated biclusters from
    gene expression data. IEEE Transactions on Knowledge and Data Engineering, 23(4), 568-584.

    :param bicluster: bicluster in which the SCS is desired
    :return: SCS score
    """

    cdef unsigned int I = bicluster.shape[0]
    cdef unsigned int J = bicluster.shape[1]
    cdef float fI = float(I), fJ = float(J)
    cdef float[:, :] bicluster_view = bicluster

    cdef float row_score
    cdef float col_score
    cdef float min_row_score = 2
    cdef float min_col_score = 2

    cdef float temp_sum = 0

    with nogil:

        for i in range(I):
            for i2 in range(I):
                if i != i2:
                    temp_sum += fabs(pearson(bicluster_view[i, :], bicluster_view[i2, :], J))
            row_score = 1 - 1/(fI-1)*temp_sum
            if row_score < min_row_score:
                min_row_score = row_score
            temp_sum = 0

        for j in range(J):
            for j2 in range(J):
                if j != j2:
                    temp_sum += fabs(pearson(bicluster_view[:, j], bicluster_view[:, j2], I))
            col_score = 1 - 1/(fJ-1)*temp_sum
            if col_score < min_col_score:
                min_col_score = col_score
            temp_sum = 0

    return min_row_score if min_row_score < min_col_score else min_col_score


cdef float pearson(float[:] x, float[:] y, unsigned int n) nogil:
    """
    Single pass pearson correlation coefficient for a sample, adapted from:
    https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
    
    :param x: first vector
    :param y: second vector
    :param n: size of vectors
    :return: pearson correlation coefficient 
    """

    cdef float sum_x = 0, sum_y = 0, sum_x2 = 0, sum_y2 = 0, sum_xy = 0

    for i in range(n):
        sum_x += x[i]
        sum_x2 += x[i]*x[i]
        sum_y += y[i]
        sum_y2 += y[i]*y[i]
        sum_xy += x[i]*y[i]

    return (n*sum_xy - sum_x*sum_y) / ((sqrt(n*sum_x2 - sum_x*sum_x)) * sqrt(n*sum_y2 - sum_y*sum_y))

