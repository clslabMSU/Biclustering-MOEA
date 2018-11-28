# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

cimport cython
from libc.stdlib cimport malloc, free


def ASR(float[:, :] bicluster):
    """
    Average Spearman's Rho for a bicluster.
    Ayadi, W., Elloumi, M., & Hao, J. K. (2009). A biclustering algorithm based on a Bicluster Enumeration Tree: application to DNA microarray data. BioData mining, 2(1), 9.

    :param bicluster: - np.ndarray of expression levels of bicluster
    :return: average spearman's rho, close to 1 or -1 is better
    """

    cdef unsigned int I = bicluster.shape[0]
    cdef unsigned int J = bicluster.shape[1]

    if I <= 1 or J <= 1:
        return 0

    cdef float asr = 0
    cdef float spearman_genes = 0
    cdef float spearman_samples = 0
    cdef unsigned int i = 0, j = 0, k = 0, l = 0

    cdef float** row_pointers = <float**>malloc(I*sizeof(float*))
    cdef float** col_pointers = <float**>malloc(J*sizeof(float*))
    cdef float* R
    cdef float[:, :] bicluster_view = bicluster

    with nogil:

        for i in range(I):
            R = <float*>malloc(J*sizeof(float*))
            rankdata(bicluster_view[i, :], R)
            row_pointers[i] = R
        for j in range(J):
            R = <float*>malloc(I*sizeof(float*))
            rankdata(bicluster_view[:, j], R)
            col_pointers[j] = R

        for i in range(I - 1):
            for j in range(i+1, I):
                spearman_genes += spearman(bicluster_view[i, :], bicluster_view[j, :], row_pointers[i], row_pointers[j])

        for k in range(J - 1):
            for l in range(k+1, J):
                spearman_samples += spearman(bicluster_view[:, k], bicluster_view[:, l], col_pointers[k], col_pointers[l])

        spearman_genes /= I*(I-1)
        spearman_samples /= J*(J-1)
        asr = 2*max(spearman_genes, spearman_samples)

        # free all malloc'd memory
        for i in range(I):
            free(row_pointers[i])
        for j in range(J):
            free(col_pointers[j])
        free(row_pointers)
        free(col_pointers)

    return asr


cdef float spearman(float[:] x, float[:] y, float* rx, float* ry) nogil:
    """
    Spearman's Rho for two vectors.
    Ayadi, W., Elloumi, M., & Hao, J. K. (2009). A biclustering algorithm based on a Bicluster Enumeration Tree: application to DNA microarray data. BioData mining, 2(1), 9.

    :param x: first vector
    :param y: second vector
    :param rx: ranks of first vector, optional
    :param ry: ranks of second vector, optional
    :return: spearman's rho between vectors, close to 1 or -1 is better
    """
    cdef int m = x.shape[0]

    # print "X: ", list(x)
    # print "RX: ",
    # for i in range(m):
    #     print rx[i],
    # print "\n",
    # print "Y: ", list(y)
    # print "RY: ",
    # for i in range(m):
    #     print ry[i],
    # print "\n"

    cdef float coef = 6.0/(m**3-m)
    cdef float ans = 0
    for k in range(m):
        ans += (rx[k] - ry[k])**2
    #print "ASR: ", 1 - coef*ans, "\n"
    return 1 - coef*ans


cdef float* rankdata(float[:] A, float* R) nogil:

    cdef int num_elems = A.shape[0]
    cdef int r = 1
    cdef int s = 1

    # Rank Vector
    for i in range(num_elems):
        R[i] = 0

    # print "A: ",
    # for i in range(num_elems):
    #     print A[i],
    # print ""

    for i in range(num_elems):
        for j in range(num_elems):
            if i == j:
                continue
            if A[j] < A[i]:
                # print str(A[j]) + " < " + str(A[i]) + "\t (i=" + str(i) + ", j=" + str(j) + ")"
                r += 1
            if A[j] == A[i]:
                # print str(A[j]) + " = " + str(A[i]) + "\t (i=" + str(i) + ", j=" + str(j) + ")"
                s += 1

        # Use formula to obtain rank
        R[i] = r + (s - 1) / 2
        r = 1
        s = 1

    # print "R: ",
    # for i in range(num_elems):
    #     print R[i],
    # print "\n"

    # Return Rank Vector
    return R