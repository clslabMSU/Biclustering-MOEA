# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

cimport cython
from libc.math cimport fabs, sqrt
from libc.stdlib cimport malloc, free


def TPC(float[:, :] bicluster):
    """
    Trend-preserving correlation score

    :param bicluster: bicluster in which the TPC is desired
    :return: TPC score
    """

    cdef unsigned int I = bicluster.shape[0]
    cdef unsigned int J = bicluster.shape[1]
    cdef float fI = float(I), fJ = float(J)
    cdef float[:, :] bicluster_view = bicluster

    cdef float total_hamming = 0
    cdef float I_choose_2 = fI*(fI-1)*0.5

    cdef unsigned int i, j = 0

    # I by J-1
    cdef float** trends = <float**>malloc(I*sizeof(float*))


    with nogil:

        for i in range(I):
            trends[i] = <float*>malloc((J-1)*sizeof(float*))
            for j in range(J-1):
                trends[i][j] = 1 if bicluster_view[i, j+1] - bicluster_view[i, j] > 0 else -1

        for i in range(I-1):
            for j in range(i+1, I):
                total_hamming += hamming_correlation(trends[i], trends[j], J)

        for i in range(I):
            free(trends[i])
        free(trends)

    return total_hamming / I_choose_2


cdef float hamming_correlation(float* x, float* y, unsigned int n) nogil:
    """    
    :param x: first vector
    :param y: second vector
    :param n: size of vectors
    :return: hamming correlation between vectors
    """
    cdef float same = 0, diff = 0
    for i in range(n):
        if x[i] == y[i]:
            same += 1
        else:
            diff += 1
    return max(same / n, diff / n)

