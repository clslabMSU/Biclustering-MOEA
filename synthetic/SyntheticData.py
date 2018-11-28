from functools import reduce
import numpy as np
from scipy.stats import rv_continuous, norm
from typing import List, Tuple, Union

from Bicluster import Bicluster
from BiclusterSet import BiclusterSet
from Dataset import Dataset

BICLUSTER_CONSTANT = 1
BICLUSTER_PATTERN = 2
DIMENSION_ROW = 4
DIMENSION_COLUMN = 8
TYPE_ADDITIVE = 16
TYPE_MULTIPLICATIVE = 32


class SyntheticData:

    def __init__(self, size: (int, int), dist: rv_continuous = norm, **distargs):
        """
        Create synthetic data of size (rows x columns) from a given distribution.
        A list of distributions can be found here:
        https://docs.scipy.org/doc/numpy/reference/routines.random.html

        :param rows: number of rows to create
        :param columns: number of samples to create
        :param dist: numpy.random routine to use (https://docs.scipy.org/doc/numpy/reference/routines.random.html)
        :param distargs: arguments to pass to scipy function, excluding size
        """
        self.rows = size[0]
        self.columns = size[1]
        self.size = size

        self.dist = dist
        self.distargs = distargs
        self.biclusters = BiclusterSet()

        # generate normalized array using given distribution
        self._array_ = dist.rvs(**distargs, size=self.size).astype(np.float32)
        self._array_ = np.divide(np.subtract(self._array_, np.min(self._array_)), np.subtract(np.max(self._array_), np.min(self._array_)))


    # TODO - overlap vs. no overlap
    # TODO - number of biclusters to generate
    # TODO - parameter for noise
    def implant_bicluster(self, size: (int, int), bicluster_type: int, **kwargs) -> Bicluster:
        """
        Implant biclusters into synthetic dataset.

        For more information on bicluster types, see Mukhopadhyay et al.'s survey:
        Mukhopadhyay, A., Maulik, U., & Bandyopadhyay, S. (2010). On biclustering of gene expression data. Current Bioinformatics, 5(3), 204-216.

        :param size: tuple of (rows, columns), size of implanted bicluster
        :param bicluster_type: combination of module level constants for bicluster type using binary OR "|".
        :param kwargs: addition parameters to pass to function. If not specified, then they are randomly chosen.
                    "genes"     - list of genes (rows) to use when implanting bicluster
                    "samples"   - list of samples (columns) to use when implanting bicluster
                    "pi"        - bicluster constant (float) as in Mukhopadhyay et al.
                    "ai"        - np.ndarray of shifting factors for row i as in Mukhopadhyay et al.
                    "bi"        - np.ndarray of scaling factors for row i as in Mukhopadhyay et al.
                    "pj"        - np.ndarray of shifting factors for column j as in Mukhopadhyay et al.
                    "qj"        - np.ndarray of scaling factors for column j as in Mukhopadhyay et al.
        :return: Bicluster object representing implanted bicluster.
        """

        # determine genes and samples in bicluster
        if "genes" in kwargs:
            assert(kwargs["genes"].shape[0] == size[0])
            genes = kwargs["genes"]
        else:
            genes = np.random.choice(self._array_.shape[0], size[0], replace=False)

        if "samples" in kwargs:
            assert(kwargs["samples"].shape[0] == size[1])
            samples = kwargs["samples"]
        else:
            samples = np.random.choice(self._array_.shape[1], size[1], replace=False)

        # constant row and/or column biclusters
        if bicluster_type & BICLUSTER_CONSTANT != 0:

            pi = kwargs["pi"] if "pi" in kwargs else np.random.uniform()
            self._array_[np.ix_(genes, samples)] = pi

            # constant rows and columns, we are done
            if bicluster_type & DIMENSION_ROW != 0 and bicluster_type & DIMENSION_COLUMN != 0:
                pass

            elif bicluster_type & DIMENSION_COLUMN != 0:
                pj = kwargs["pj"] if "pj" in kwargs else np.random.uniform(size=(1, size[1]))
                qj = kwargs["qj"] if "qj" in kwargs else np.random.uniform(size=(1, size[1]))
                if bicluster_type & TYPE_MULTIPLICATIVE != 0:
                    self._array_[np.ix_(genes, samples)] = np.multiply(self._array_[np.ix_(genes, samples)], qj)
                if bicluster_type & TYPE_ADDITIVE != 0:
                    self._array_[np.ix_(genes, samples)] = np.add(self._array_[np.ix_(genes, samples)], pj)
                if bicluster_type & (TYPE_MULTIPLICATIVE | TYPE_ADDITIVE) == 0:
                    raise ValueError("Bicluster has no type: additive or multiplicative.")

            elif bicluster_type & DIMENSION_ROW != 0:
                ai = kwargs["ai"] if "ai" in kwargs else np.random.uniform(size=(size[0], 1))
                bi = kwargs["bi"] if "bi" in kwargs else np.random.uniform(size=(size[0], 1))
                if bicluster_type & TYPE_MULTIPLICATIVE != 0:
                    self._array_[np.ix_(genes, samples)] = np.multiply(self._array_[np.ix_(genes, samples)], bi)
                if bicluster_type & TYPE_ADDITIVE != 0:
                    self._array_[np.ix_(genes, samples)] = np.add(self._array_[np.ix_(genes, samples)], ai)
                if bicluster_type & (TYPE_MULTIPLICATIVE | TYPE_ADDITIVE) == 0:
                    # TODO - don't throw error on constant bicluster with no type
                    raise ValueError("Bicluster has no type: additive or multiplicative.")
            else:
                raise ValueError("Invalid configuration of bicluster type.")

        elif bicluster_type & BICLUSTER_PATTERN != 0:

            if bicluster_type & (TYPE_MULTIPLICATIVE | TYPE_ADDITIVE) == 0:
                raise ValueError("Bicluster has no type: additive or multiplicative.")

            pi = kwargs["pi"] if "pi" in kwargs else np.random.uniform()
            self._array_[np.ix_(genes, samples)] = pi

            if bicluster_type & TYPE_MULTIPLICATIVE != 0:
                bi = kwargs["bi"] if "bi" in kwargs else np.random.uniform(size=(size[0], 1))
                qj = kwargs["qj"] if "qj" in kwargs else np.random.uniform(size=(1, size[1]))
                self._array_[np.ix_(genes, samples)] = np.multiply(self._array_[np.ix_(genes, samples)], bi)
                self._array_[np.ix_(genes, samples)] = np.multiply(self._array_[np.ix_(genes, samples)], qj)

            if bicluster_type & TYPE_ADDITIVE != 0:
                ai = kwargs["ai"] if "ai" in kwargs else np.random.uniform(size=(size[0], 1))
                pj = kwargs["pj"] if "pj" in kwargs else np.random.uniform(size=(1, size[1]))
                self._array_[np.ix_(genes, samples)] = np.add(self._array_[np.ix_(genes, samples)], ai)
                self._array_[np.ix_(genes, samples)] = np.add(self._array_[np.ix_(genes, samples)], pj)

        bicluster = Bicluster(genes, samples, self._array_[np.ix_(genes, samples)])
        self.biclusters.append(bicluster)
        return bicluster


    def implant_trend_preserving_bicluster(self, size: (int, int), noise_std: float,
                                           indices: Union[Tuple[List[int], List[int]], None] = None,
                                           allow_overlap: bool = False, symmetric: bool = False) -> Bicluster:
        """
        Implant an approximately trend-preserving bicluster in the dataset of given size. Normally-distributed noise is
        added to the generated bicluster from a normal distribution with zero mean and noise_std standard deviation.

        Large values for noise_std will cause the ranks of rows in the bicluster to change, making it difficult to
        find these biclusters.

        :param size: size of bicluster to be implanted
        :param noise_std: standard deviation of normally distributed noise, 0 for no noise
        :param indices: optional indices to specify where bicluster is planted
        :param allow_overlap: whether biclusters are allowed to overlap
        :param symmetric: whether biclusters should be symmetric (trend*constant) or not (trend+random)
        :return: new bicluster object
        """

        if indices is not None:
            assert(len(indices[0]) == size[0])
            assert(len(indices[1]) == size[1])
        else:
            if allow_overlap or len(self.biclusters) == 0:
                indices = (np.random.choice(range(self._array_.shape[0]), size=size[0], replace=False),
                           np.random.choice(range(self._array_.shape[1]), size=size[1], replace=False))
            else:
                used_rows = np.array(list(reduce(lambda x,y : x+y, list(map(lambda bicluster:bicluster.genes(), self.biclusters)))))
                used_cols = np.array(list(reduce(lambda x,y : x+y, list(map(lambda bicluster:bicluster.samples(), self.biclusters)))))
                indices = (np.random.choice(list(filter(lambda x:x not in used_rows, range(self._array_.shape[0]))), size=size[0], replace=False),
                           np.random.choice(list(filter(lambda x:x not in used_cols, range(self._array_.shape[1]))), size=size[1], replace=False))

        # initialize bicluster to empty
        bic_values = np.empty(size, dtype=np.float32)

        # find endpoints of random centered at median such that about 95% of data is within
        rand_min, rand_max = self.dist.interval(0.95)

        if symmetric:
            # make trend
            trend = np.arange(1, size[1] + 1)
            np.random.shuffle(trend)
            scaled_trend = np.divide(trend, size[1])

            # add each row
            for i in range(size[0]):

                # generate noise to add to bicluster
                noise = np.random.normal(0, noise_std, size=size[1])

                # randomly choose sign to be positive or negative
                sign = [-1, 1][np.random.randint(0, 2)]

                # uniformly choose scale such that trend falls within about 95% of the data
                scale = np.random.uniform(rand_min, rand_max)

                # calculate actual row of bicluster
                bic_values[i, :] = sign * scale * scaled_trend + noise

        else:

            # generate initial pattern of ups (1) and downs (-1)
            pattern = np.random.randint(0, 2, size=size[1])*2 - 1

            for i in range(size[0]):

                # initialize row to zeros
                row = np.zeros(size[1])

                # pick sign, 1 indicating order-preserving row and -1 indicating order-reversing row
                sign = [-1, 1][np.random.randint(0, 2)]

                # sample from chi-square(3) to get first entry
                row[0] = np.random.chisquare(3)

                # add random chi-square(3) value with appropriate sign to previous entry, adding Gaussian noise.
                for j in range(1, size[1]):
                    row[j] = row[j-1] + sign*pattern[j]*np.random.chisquare(3) + np.random.normal(0, noise_std)

                # scale row back into range [rand_min, rand_max]
                bic_values[i, :] = np.multiply(np.divide(row - min(row), max(row) - min(row)), rand_max - rand_min) + rand_min


        self._array_[np.ix_(indices[0], indices[1])] = bic_values
        bic = Bicluster(indices[0], indices[1], bic_values)
        self.biclusters.append(bic)
        return bic


    def to_dataset(self, name: str, filename: Union[str, None] = None, gene_labels: Union[List[str], None] = None, sample_labels: Union[List[str], None] = None) -> Dataset:
        """
        Convert SyntheticData object into Dataset object.

        :param name: name of dataset
        :param filename: filename to save as, defaults to "name.txt"
        :param gene_labels: defaults to "g0, g1, ..."
        :param sample_labels: defaults to "c0, c1, ..."
        :return: Dataset object
        """
        if filename is None:
            filename = name + ".txt"
        if gene_labels is None:
            gene_labels = ["g" + str(i) for i in range(self._array_.shape[0])]
        if sample_labels is None:
            sample_labels = ["c" + str(j) for j in range(self._array_.shape[1])]

        return Dataset(name, filename, gene_labels, sample_labels, self._array_, self.biclusters)


    def array(self):
        return self._array_


    def __len__(self):
        return self.rows * self.columns


if __name__ == "__main__":


    # narrow datasets
    data_sizes = [(1000, 100)]
    bic_sizes = [(150, 15), (100, 20), (100, 25)]
    num_bics = 3
    reps = 5
    #bic_indices = [(list(range(300)), list(range(30))), (list(range(300, 600)), list(range(30, 60))), (list(range(600, 900)), list(range(60, 90))), None, None]
    for data_size in data_sizes:
        for bic_size in bic_sizes:
            for rep in range(reps):
                data = SyntheticData(data_size)
                bics = BiclusterSet()
                for i in range(num_bics):
                    bics.append(data.implant_trend_preserving_bicluster(bic_size, 0, (list(range(i*bic_size[0], (i+1)*bic_size[0])), list(range(i*bic_size[1], (i+1)*bic_size[1])))))
                dataset = data.to_dataset("narrow_(%d,%d)_(%d,%d)_typeI_data%s" % (data_size[0], data_size[1], bic_size[0], bic_size[1], rep+1))
                dataset.known_bics = bics
                dataset.write_unibic_format()

    # square datasets
    data_sizes = [(150, 100), (200, 150), (300, 200)]
    bic_sizes = [(15, 15), (20, 20), (25, 25)]
    num_bics = [3, 4, 5]
    reps = 5

    for i in range(len(data_sizes)):
        for rep in range(reps):
            data = SyntheticData(data_sizes[i])
            bics = BiclusterSet()
            for k in range(num_bics[i]):
                bics.append(data.implant_trend_preserving_bicluster(bic_sizes[i], 0, (list(range(k*bic_sizes[i][0], (k+1)*bic_sizes[i][0])), list(range(k*bic_sizes[i][1], (k+1)*bic_sizes[i][1])))))
            dataset = data.to_dataset("square_(%d,%d)_(%d,%d)_typeI_data%s" % (data_sizes[i][0], data_sizes[i][1], bic_sizes[i][0], bic_sizes[i][1], rep+1))
            dataset.known_bics = bics
            dataset.write_unibic_format()
