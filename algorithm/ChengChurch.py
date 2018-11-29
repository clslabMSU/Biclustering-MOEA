"""
Implements the Cheng and Church biclustering algorithm.

Original Author : Kemal Eren (https://github.com/kemaleren)
License: BSD 3 clause

Separated from outdated branch of scikit-learn for portability.

"""

from abc import ABCMeta

import numpy as np
from sklearn.base import BaseEstimator, BiclusterMixin
from sklearn.externals import six
from sklearn.utils.validation import check_random_state
from sys import stdout

from Bicluster import Bicluster
from BiclusterSet import BiclusterSet
from Dataset import Dataset

from algorithm._ChengChurch.utils import check_array_ndim, check_arrays

try:
    from algorithm._ChengChurch.mean_squared_residue import mean_squared_residue
except ImportError:
    print("Building mean_squared_residue function... This should only happen once.")
    from os import getcwd, system
    from shutil import rmtree
    from sys import executable
    build_path = getcwd().replace(" ", "\\ ") + "/algorithm/_ChengChurch"
    command = "{python} {path}/build.py build_ext --inplace >/dev/null".format(python=executable, path=build_path)
    print("Executing command: " + command)
    system(command)
    from algorithm._ChengChurch.mean_squared_residue import mean_squared_residue
    print("Removing build files...")
    rmtree("build")
    print("Finished build.")


try:
    from numpy import count_nonzero
except ImportError:
    def count_nonzero(x):
        return (x != 0).sum()


class EmptyBiclusterException(Exception):
    pass


class _IncrementalMSR(object):
    """Incrementally calculates MSR during node deletion."""
    def __init__(self, rows, cols, arr, tol=1e-5):
        assert rows.dtype == np.bool
        assert cols.dtype == np.bool

        self.arr = arr
        self.rows = rows
        self.cols = cols
        self.tol = tol

        self._row_idxs = None
        self._col_idxs = None

        subarr = arr[self.get_row_idxs()[:, np.newaxis], self.get_col_idxs()]
        self._sum = subarr.sum()
        self._row_sum = subarr.sum(axis=1)
        self._col_sum = subarr.sum(axis=0)

        self._reset()

    def _reset(self):
        self._msr = None
        self._row_msr = None
        self._col_msr = None

    def get_row_idxs(self):
        if self._row_idxs is None:
            self._row_idxs = np.nonzero(self.rows)[0]
        return self._row_idxs

    def get_col_idxs(self):
        if self._col_idxs is None:
            self._col_idxs = np.nonzero(self.cols)[0]
        return self._col_idxs

    def remove_row(self, row):
        if not self.rows[row]:
            raise ValueError('cannot remove row {}; it is not in the'
                             ' bicluster'.format(row))
        if len(self.get_row_idxs()) <= 1:
            raise EmptyBiclusterException()
        self._reset()
        vec = self.arr[row, self.get_col_idxs()].ravel()
        self._sum -= vec.sum()
        self._col_sum -= vec

        idx = np.searchsorted(self.get_row_idxs(), row)
        self._row_sum = np.delete(self._row_sum, idx)
        self.rows[row] = False
        self._row_idxs = None

    def remove_col(self, col):
        if not self.cols[col]:
            raise ValueError('cannot remove col {}; it is not in the'
                             ' bicluster'.format(col))
        if len(self.get_col_idxs()) <= 1:
            raise EmptyBiclusterException()
        self._reset()
        vec = self.arr[self.get_row_idxs(), col].ravel()
        self._sum -= vec.sum()
        self._row_sum -= vec

        idx = np.searchsorted(self.get_col_idxs(), col)
        self._col_sum = np.delete(self._col_sum, idx)
        self.cols[col] = False
        self._col_idxs = None

    def remove_rows(self, rows):
        for r in rows:
            self.remove_row(r)

    def remove_cols(self, cols):
        for c in cols:
            self.remove_col(c)

    def _compute(self):
        n_rows = len(self.get_row_idxs())
        n_cols = len(self.get_col_idxs())

        row_mean = self._row_sum / n_cols
        col_mean = self._col_sum / n_rows
        mean = self._sum / (n_rows * n_cols)

        self._msr, self._row_msr, self._col_msr = \
            mean_squared_residue(self.get_row_idxs(),
                                 self.get_col_idxs(), row_mean,
                                 col_mean, mean, self.arr)
        self._msr = 0 if self._msr < self.tol else self._msr
        self._row_msr[self._row_msr < self.tol] = 0
        self.col_msr[self._col_msr < self.tol] = 0

    @property
    def msr(self):
        if self._msr is None:
            self._compute()
        return self._msr

    @property
    def row_msr(self):
        if self._row_msr is None:
            self._compute()
        return self._row_msr

    @property
    def col_msr(self):
        if self._col_msr is None:
            self._compute()
        return self._col_msr


class ChengChurch(six.with_metaclass(ABCMeta, BaseEstimator,
                                     BiclusterMixin)):
    """Algorithm that finds biclusters with small mean squared residue (MSR).

    The residue of an array ``X`` is calculated as ``X -
    X.mean(axis=1, keepdims=True) - X.mean(axis=0) + X.mean()``. It measures
    each element's coherence with the overall mean, row mean, and column
    mean. To get the mean squared residue, the residues are squared and their
    mean is calculated.

    ChengChurch tries to maximize bicluser size with the constraint
    that its mean squared residue cannot exceed ``max_msr``.

    Parameters
    -----------
    n_clusters : integer, optional, default: 100
        The number of biclusters to find.

    max_msr : float, default: 1.0
        Maximum mean squared residue of a bicluster. Equivalent to
        'delta` in original paper.

    deletion_threshold : float, optional, default: 1.5
        Multiplier for multiple node deletion. Equivalent to `alpha`
        in original paper.

    row_deletion_cutoff : integer, optional, default: 100
        Number of rows at which to switch to single node deletion.

    column_deletion_cutoff : integer, optional, default: 100
        Number of columns at which to switch to single node deletion.

    inverse_rows : bool, optional, default: False
        During node addition, add rows if their inverse has a low MSR.

    random_state : int seed, RandomState instance, or None (default)
        A pseudo random number generator used by the K-Means
        initialization.

    Attributes
    ----------
    `rows_` : array-like, shape (n_row_clusters, n_rows)
        Results of the clustering. `rows[i, r]` is True if cluster `i`
        contains row `r`. Available only after calling ``fit()``.

    `columns_` : array-like, shape (n_column_clusters, n_columns)
        Results of the clustering, like `rows`.

    `inverted_rows` : array-like, shape (n_row_clusters, n_rows)
        `inverted_rows[i, r` is True if row `r` was inverted to match
        the pattern of cluster `i`.


    References
    ----------

    - Cheng, Y., & Church, G. M. (2000). `Biclustering of
      expression data
      <ftp://samba.ad.sdsc.edu/pub/sdsc/biology/ISMB00/157.pdf>`__.

    """

    def __init__(self, n_clusters=100, max_msr=1.0,
                 deletion_threshold=1.5, row_deletion_cutoff=100,
                 column_deletion_cutoff=100, inverse_rows=False,
                 random_state=None):
        self.n_clusters = n_clusters
        self.max_msr = max_msr
        self.deletion_threshold = deletion_threshold
        self.row_deletion_cutoff = row_deletion_cutoff
        self.column_deletion_cutoff = column_deletion_cutoff
        self.inverse_rows = inverse_rows
        self.random_state = random_state


    def _check_parameters(self):
        if self.n_clusters < 1:
            raise ValueError("'n_clusters' must be > 0, but its value"
                             " is {}".format(self.n_clusters))
        if self.max_msr < 0:
            raise ValueError("'max_msr' must be > 0.0, but its value"
                             " is {}".format(self.max_msr))
        if self.deletion_threshold < 1:
            raise ValueError("'deletion_threshold' must be >= 1.0, but its"
                             " value is {}".format(self.deletion_threshold))
        if self.row_deletion_cutoff < 1:
            raise ValueError("'row_deletion_cutoff' must be >= 1, but its"
                             " value is {}".format(self.row_deletion_cutoff))
        if self.column_deletion_cutoff < 1:
            raise ValueError("'column_deletion_cutoff' must be >= 1, but its"
                             " value is {}".format(
                                 self.column_deletion_cutoff))


    def _msr(self, rows, cols, X):
        """Compute the MSR of a bicluster."""
        rows = rows.nonzero()[0][:, np.newaxis]
        cols = cols.nonzero()[0]
        if not rows.size or not cols.size:
            raise EmptyBiclusterException()
        sub = X[rows, cols]
        residue = (sub - sub.mean(axis=1, keepdims=True) -
                   sub.mean(axis=0) + sub.mean())
        return np.power(residue, 2).mean()


    def _row_msr(self, rows, cols, X, inverse=False):
        """Compute MSR of all rows for adding them to the bicluster."""
        if not count_nonzero(rows) or not count_nonzero(cols):
            raise EmptyBiclusterException()
        row_mean = X[:, cols].mean(axis=1, keepdims=True)
        col_mean = X[rows][:, cols].mean(axis=0)
        if inverse:
            arr = (-X[:, cols] + row_mean - col_mean +
                   X[rows][:, cols].mean())
        else:
            arr = (X[:, cols] - row_mean - col_mean +
                   X[rows][:, cols].mean())
        return np.power(arr, 2).mean(axis=1)


    def _col_msr(self, rows, cols, X):
        """Compute MSR of all columns for adding them to the bicluster."""
        if not rows.size or not cols.size:
            raise EmptyBiclusterException()
        row_mean = X[rows][:, cols].mean(axis=1, keepdims=True)
        col_mean = X[rows, :].mean(axis=0)
        arr = X[rows, :] - row_mean - col_mean + X[rows][:, cols].mean()
        return np.power(arr, 2).mean(axis=0)


    def _single_node_deletion(self, rows, cols, X):
        """Iteratively remove single rows and columns."""
        inc = _IncrementalMSR(rows, cols, X)
        while inc.msr > self.max_msr:
            row_idx = np.argmax(inc.row_msr)
            col_idx = np.argmax(inc.col_msr)
            if inc.row_msr[row_idx] > inc.col_msr[col_idx]:
                inc.remove_row(inc.get_row_idxs()[row_idx])
            else:
                inc.remove_col(inc.get_col_idxs()[col_idx])
        return inc.rows, inc.cols


    def _multiple_node_deletion(self, rows, cols, X):
        """Iteratively remove multiple rows and columns at once."""
        inc = _IncrementalMSR(rows, cols, X)
        while inc.msr > self.max_msr:
            n_rows = len(inc.get_row_idxs())
            n_cols = len(inc.get_col_idxs())
            if n_rows >= self.row_deletion_cutoff:
                to_remove = inc.row_msr > (self.deletion_threshold * inc.msr)
                inc.remove_rows(inc.get_row_idxs()[to_remove])

            if n_cols >= self.column_deletion_cutoff:
                to_remove = inc.col_msr > (self.deletion_threshold *
                                           inc.msr)
                inc.remove_cols(inc.get_col_idxs()[to_remove])

            if (n_rows == len(inc.get_row_idxs()) and
                n_cols == len(inc.get_col_idxs())):
                break
        return inc.rows, inc.cols


    def _node_addition(self, rows, cols, X):
        """Add rows and columns with MSR smaller than the bicluster's."""
        inverse_rows = np.zeros(len(rows), dtype=np.bool)
        while True:
            n_rows = count_nonzero(rows)
            n_cols = count_nonzero(cols)

            msr = self._msr(rows, cols, X)
            col_msr = self._col_msr(rows, cols, X)
            cols = np.logical_or(cols, col_msr < msr)

            old_rows = rows.copy()  # save for row inverse
            msr = self._msr(rows, cols, X)
            row_msr = self._row_msr(rows, cols, X)
            rows = np.logical_or(rows, row_msr < msr)

            if self.inverse_rows:
                row_msr = self._row_msr(old_rows, cols, X,
                                        inverse=True)
                to_add = row_msr < msr
                new_inverse_rows = np.logical_and(to_add, np.logical_not(rows))
                inverse_rows = np.logical_or(inverse_rows,
                                             new_inverse_rows)
                rows = np.logical_or(rows, to_add)

            if (n_rows == count_nonzero(rows)) and \
               (n_cols == count_nonzero(cols)):
                break
        return rows, cols, inverse_rows


    def _mask(self, X, rows, cols, generator, minval, maxval):
        """Mask a bicluster in the data with random values."""
        shape = count_nonzero(rows), count_nonzero(cols)
        mask_vals = generator.uniform(minval, maxval, shape)
        r = rows.nonzero()[0][:, np.newaxis]
        c = cols.nonzero()[0]
        X[r, c] = mask_vals


    def fit(self, X: np.ndarray) -> 'ChengChurch':
        """Creates a biclustering for X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        """
        self._check_parameters()
        X, = check_arrays(X, copy=True, dtype=np.float64)
        check_array_ndim(X)
        minval, maxval = X.min(), X.max()
        n_rows, n_cols = X.shape

        generator = check_random_state(self.random_state)
        result_rows = []
        result_cols = []
        inverse_rows = []

        for i in range(self.n_clusters):
            rows = np.ones(n_rows, dtype=np.bool)
            cols = np.ones(n_cols, dtype=np.bool)
            rows, cols = self._multiple_node_deletion(rows, cols, X)
            rows, cols = self._single_node_deletion(rows, cols, X)
            rows, cols, irows = self._node_addition(rows, cols, X)
            if rows.sum() == 1 or cols.sum() == 1:
                break  # trivial bicluster
            self._mask(X, rows, cols, generator, minval, maxval)
            result_rows.append(rows)
            result_cols.append(cols)
            inverse_rows.append(irows)

        if len(result_rows) > 0:
            self.rows_ = np.vstack(result_rows)
            self.columns_ = np.vstack(result_cols)
            self.inverted_rows = np.vstack(inverse_rows)
        else:
            self.rows_ = np.zeros((0, n_rows))
            self.columns_ = np.zeros((0, n_cols))
            self.inverted_rows = np.zeros((0, n_rows))

        return self

    @staticmethod
    def cluster(dataset: Dataset, **kwargs):
        stdout.write("\r\t\tRunning ChengChurch..."); stdout.flush()
        self = ChengChurch(**kwargs)
        X = dataset.matrix
        return self.fit(X).toBiclusterSet(X)


    def toBiclusterSet(self, dataset: np.ndarray) -> BiclusterSet:
        biclusters = []
        for i in range(len(self.rows_)):
            stdout.write("\r\t\tComputing bicluster %d/%d (%.2f%%)" % (i+1, len(self.rows_), 100 * float(i+1) / len(self.rows_))); stdout.flush()
            genes, samples = self.get_indices(i)
            biclusters.append(Bicluster(genes, samples, self.get_submatrix(i, dataset)))
        stdout.write("\r"); stdout.flush()
        return BiclusterSet(biclusters)



if __name__ == "__main__":
    from urllib.request import urlopen
    import numpy as np
    from pandas import DataFrame
    from pandas.tools.plotting import parallel_coordinates
    import matplotlib.pylab as plt

    # get data
    url = "http://arep.med.harvard.edu/biclustering/lymphoma.matrix"
    lines = urlopen(url).read().decode().strip().split('\n')
    # insert a space before all negative signs
    lines = list(' -'.join(line.split('-')).split(' ') for line in lines)
    lines = list(list(int(i) for i in line if i) for line in lines)
    data = np.array(lines)

    # replace missing values, just as in the paper
    generator = np.random.RandomState(0)
    idx = np.where(data == 999)
    data[idx] = generator.randint(-800, 801, len(idx[0]))

    # cluster with same parameters as original paper
    model = ChengChurch(n_clusters=100, max_msr=1200,
                        deletion_threshold=1.2, inverse_rows=True,
                        random_state=0)
    model.fit(data)

    # find bicluster with smallest msr and plot it
    msr = lambda a: (np.power(a - a.mean(axis=1, keepdims=True) -
                              a.mean(axis=0) + a.mean(), 2).mean())
    msrs = list(msr(model.get_submatrix(i, data)) for i in range(100))
    arr = model.get_submatrix(np.argmin(msrs), data)
    df = DataFrame(arr)
    df['row'] = map(str, range(arr.shape[0]))
    parallel_coordinates(df, 'row', linewidth=1.5)
    plt.xlabel('column')
    plt.ylabel('expression level')
    plt.gca().legend_ = None
    plt.show()