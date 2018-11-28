import numpy as np
import unittest

from synthetic.SyntheticData import SyntheticData
from synthetic.SyntheticData import BICLUSTER_CONSTANT, BICLUSTER_PATTERN
from synthetic.SyntheticData import DIMENSION_ROW, DIMENSION_COLUMN
from synthetic.SyntheticData import TYPE_ADDITIVE, TYPE_MULTIPLICATIVE

from Metrics import VEt


def isConstantBicluster(bicluster: np.ndarray) -> bool:
    return np.all(bicluster == bicluster[0, 0])

# noinspection PyTypeChecker
def isRowConstantBicluster(bicluster: np.ndarray) -> bool:
    for gene in range(bicluster.shape[0]):
        return np.all(bicluster[gene, :] == bicluster[gene, 0])

# noinspection PyTypeChecker
def isColumnConstantBicluster(bicluster: np.ndarray) -> bool:
    for sample in range(bicluster.shape[1]):
        return np.all(bicluster[:, sample] == bicluster[0, sample])

# noinspection PyTypeChecker
def isAdditivePatternBicluster(bicluster: np.ndarray, tol=10e-5) -> bool:
    for sample in range(bicluster.shape[1] - 1):
        difference = bicluster[:, sample] - bicluster[:, sample + 1]
        if not np.all(np.abs(difference - difference[0]) < tol):
            return False
    return True

# noinspection PyTypeChecker
def isMultiplicativePatternBicluster(bicluster: np.ndarray, tol=10e-3) -> bool:
    for sample in range(bicluster.shape[1] - 1):
        quotient = np.divide(bicluster[:, sample + 1], bicluster[:, sample])
        if not np.all(np.abs(quotient - quotient[0]) < tol):
            return False
    return True

def isAdditiveAndMultiplicativeBicluster(bicluster: np.ndarray, tol=10e-5) -> bool:
    # check if transposed virtual error is close to 1, TODO - fix? lazy but funny
    return abs(VEt(bicluster)) - 1 < tol


# noinspection PyTypeChecker
class SyntheticDataTests(unittest.TestCase):

    def testConstantBicluster(self):

        # test case 1 - 5x5 synthetic data with 3x3 constant bicluster (no bicluster type)
        data = SyntheticData((5, 5))
        bicluster = data.implant_bicluster((3, 3), bicluster_type = BICLUSTER_CONSTANT | DIMENSION_ROW | DIMENSION_COLUMN)
        self.assertTrue(isConstantBicluster(data._array_[np.ix_(bicluster.genes(), bicluster.samples())]))

    def testConstantBicluster2(self):

        # test case 2 - 50x50 synthetic data with 7x7 constant bicluster (additive type)
        data = SyntheticData((50, 50))
        bicluster = data.implant_bicluster((7, 7), bicluster_type = BICLUSTER_CONSTANT | DIMENSION_ROW | DIMENSION_COLUMN | TYPE_ADDITIVE)
        self.assertTrue(isConstantBicluster(data._array_[np.ix_(bicluster.genes(), bicluster.samples())]))

    def testConstantBicluster3(self):

        # test case 3 - 11391x72 synthetic data with 238x19 constant bicluster (multiplicative type)
        data = SyntheticData((11391, 72))
        bicluster = data.implant_bicluster((238, 19), bicluster_type = BICLUSTER_CONSTANT | DIMENSION_ROW | DIMENSION_COLUMN | TYPE_MULTIPLICATIVE)
        self.assertTrue(isConstantBicluster(data._array_[np.ix_(bicluster.genes(), bicluster.samples())]))

    def testConstantBicluster4(self):

        # test case 4 - 11391x72 synthetic data with 238x19 constant bicluster (additive and multiplicative type)
        data = SyntheticData((11391, 72))
        bicluster = data.implant_bicluster((238, 19), bicluster_type = BICLUSTER_CONSTANT | DIMENSION_ROW | DIMENSION_COLUMN | TYPE_ADDITIVE | TYPE_MULTIPLICATIVE)
        self.assertTrue(isConstantBicluster(data._array_[np.ix_(bicluster.genes(), bicluster.samples())]))

    def testConstantBicluster5(self):

        # test case 5 - 11391x72 synthetic data with manually specified constant bicluster
        data = SyntheticData((11391, 72))
        implanted_genes = np.asarray([1642, 326, 3548, 3, 6336])
        implanted_samples = np.asarray([1, 6, 8, 3, 66])
        bicluster = data.implant_bicluster(
            size = (5, 5),
            pi = 3.14159,
            genes = implanted_genes,
            samples = implanted_samples,
            bicluster_type = BICLUSTER_CONSTANT | DIMENSION_ROW | DIMENSION_COLUMN | TYPE_ADDITIVE | TYPE_MULTIPLICATIVE
        )
        self.assertTrue(np.all(data._array_[np.ix_(implanted_genes, implanted_samples)] == 3.14159))
        self.assertTrue(sorted(bicluster.genes()) == sorted(implanted_genes))
        self.assertTrue(sorted(bicluster.samples()) == sorted(implanted_samples))

    def testRowConstantBicluster(self):

        # test case 1 - 5x5 synthetic data with 3x3 row constant bicluster (no bicluster type)
        data = SyntheticData((5, 5))
        with self.assertRaises(ValueError):
            data.implant_bicluster((3, 3), bicluster_type = BICLUSTER_CONSTANT | DIMENSION_ROW)

    def testRowConstantBicluster2(self):

        # test case 2 - 50x50 synthetic data with 7x7 row constant bicluster (additive type)
        data = SyntheticData((50, 50))
        bicluster = data.implant_bicluster((7, 7), bicluster_type = BICLUSTER_CONSTANT | DIMENSION_ROW | TYPE_ADDITIVE)
        self.assertTrue(isRowConstantBicluster(data._array_[np.ix_(bicluster.genes(), bicluster.samples())]))

    def testRowConstantBicluster3(self):

        # test case 3 - 11391x72 synthetic data with 238x19 row constant bicluster (multiplicative type)
        data = SyntheticData((11391, 72))
        bicluster = data.implant_bicluster((238, 19), bicluster_type = BICLUSTER_CONSTANT | DIMENSION_ROW | TYPE_MULTIPLICATIVE)
        self.assertTrue(isRowConstantBicluster(data._array_[np.ix_(bicluster.genes(), bicluster.samples())]))

    def testRowConstantBicluster4(self):

        # test case 4 - 11391x72 synthetic data with 238x19 row constant bicluster (additive and multiplicative type)
        data = SyntheticData((11391, 72))
        bicluster = data.implant_bicluster((238, 19), bicluster_type = BICLUSTER_CONSTANT | DIMENSION_ROW | TYPE_ADDITIVE | TYPE_MULTIPLICATIVE)
        self.assertTrue(isRowConstantBicluster(data._array_[np.ix_(bicluster.genes(), bicluster.samples())]))

    def testRowConstantBicluster5(self):

        # test case 5 - 11391x72 synthetic data with manually specified row constant bicluster
        data = SyntheticData((11391, 72))
        implanted_genes = np.asarray([1642, 326, 3548, 3, 6336])
        implanted_samples = np.asarray([1, 6, 8, 3, 66])
        pi, ai, bi = 3.14159, np.random.uniform(size=(5, 1)), np.random.uniform(size=(5, 1))
        bicluster = data.implant_bicluster(
            size = (5, 5),
            pi = pi,
            ai = ai,
            bi = bi,
            genes = implanted_genes,
            samples = implanted_samples,
            bicluster_type = BICLUSTER_CONSTANT | DIMENSION_ROW | TYPE_ADDITIVE | TYPE_MULTIPLICATIVE
        )
        for gene in range(len(bicluster.genes())):
            self.assertTrue(np.all(data._array_[bicluster.genes()[gene], bicluster.samples()] == data._array_[bicluster.genes()[gene], bicluster.samples()[0]]))
            self.assertTrue(np.allclose(data._array_[bicluster.genes()[gene], bicluster.samples()], pi*bi[gene] + ai[gene]))
        self.assertTrue(sorted(bicluster.genes()) == sorted(implanted_genes))
        self.assertTrue(sorted(bicluster.samples()) == sorted(implanted_samples))

    def testColumnConstantBicluster(self):

        # test case 1 - 5x5 synthetic data with 3x3 column constant bicluster (no bicluster type)
        data = SyntheticData((5, 5))
        with self.assertRaises(ValueError):
            data.implant_bicluster((3, 3), bicluster_type = BICLUSTER_CONSTANT | DIMENSION_COLUMN)

    def testColumnConstantBicluster2(self):

        # test case 2 - 50x50 synthetic data with 7x7 column constant bicluster (additive type)
        data = SyntheticData((50, 50))
        bicluster = data.implant_bicluster((7, 7), bicluster_type = BICLUSTER_CONSTANT | DIMENSION_COLUMN | TYPE_ADDITIVE)
        self.assertTrue(isColumnConstantBicluster(data._array_[np.ix_(bicluster.genes(), bicluster.samples())]))

    def testColumnConstantBicluster3(self):

        # test case 3 - 11391x72 synthetic data with 238x19 column constant bicluster (multiplicative type)
        data = SyntheticData((11391, 72))
        bicluster = data.implant_bicluster((238, 19), bicluster_type = BICLUSTER_CONSTANT | DIMENSION_COLUMN | TYPE_MULTIPLICATIVE)
        self.assertTrue(isColumnConstantBicluster(data._array_[np.ix_(bicluster.genes(), bicluster.samples())]))

    def testColumnConstantBicluster4(self):

        # test case 4 - 11391x72 synthetic data with 238x19 column constant bicluster (additive and multiplicative type)
        data = SyntheticData((11391, 72))
        bicluster = data.implant_bicluster((238, 19), bicluster_type = BICLUSTER_CONSTANT | DIMENSION_COLUMN | TYPE_ADDITIVE | TYPE_MULTIPLICATIVE)
        self.assertTrue(isColumnConstantBicluster(data._array_[np.ix_(bicluster.genes(), bicluster.samples())]))

    def testColumnConstantBicluster5(self):

        # test case 5 - 11391x72 synthetic data with manually specified column constant bicluster
        data = SyntheticData((11391, 72))
        implanted_genes = np.asarray([1642, 326, 3548, 3, 6336])
        implanted_samples = np.asarray([1, 6, 8, 3, 66])
        pi, qj, pj = 3.14159, np.random.uniform(size=(1, 5)), np.random.uniform(size=(1, 5))
        bicluster = data.implant_bicluster(
            size = (5, 5),
            pi = pi,
            qj = qj,
            pj = pj,
            genes = implanted_genes,
            samples = implanted_samples,
            bicluster_type = BICLUSTER_CONSTANT | DIMENSION_COLUMN | TYPE_ADDITIVE | TYPE_MULTIPLICATIVE
        )
        for sample in range(len(bicluster.samples())):
            self.assertTrue(np.all(data._array_[bicluster.genes(), bicluster.samples()[sample]] == data._array_[bicluster.genes()[0], bicluster.samples()[sample]]))
            self.assertTrue(np.allclose(data._array_[bicluster.genes(), bicluster.samples()[sample]], pi*qj[0, sample] + pj[0, sample]))
        self.assertTrue(sorted(bicluster.genes()) == sorted(implanted_genes))
        self.assertTrue(sorted(bicluster.samples()) == sorted(implanted_samples))

    def testPatternBicluster(self):

        # test case 1 - 5x5 synthetic data with 3x3 column pattern bicluster (no bicluster type)
        data = SyntheticData((5, 5))
        with self.assertRaises(ValueError):
            data.implant_bicluster((3, 3), bicluster_type = BICLUSTER_PATTERN)

    def testPatternBicluster2(self):

        # test case 2 - 50x50 synthetic data with 7x7 pattern bicluster (additive type)
        data = SyntheticData((50, 50))
        bicluster = data.implant_bicluster((7, 7), bicluster_type = BICLUSTER_PATTERN | TYPE_ADDITIVE)
        self.assertTrue(isAdditivePatternBicluster(data._array_[np.ix_(bicluster.genes(), bicluster.samples())]))

    def testPatternBicluster3(self):

        # test case 3 - 11391x72 synthetic data with 238x19 pattern bicluster (multiplicative type)
        data = SyntheticData((11391, 72))
        bicluster = data.implant_bicluster((238, 19), bicluster_type = BICLUSTER_PATTERN | TYPE_MULTIPLICATIVE)
        self.assertTrue(isMultiplicativePatternBicluster(data._array_[np.ix_(bicluster.genes(), bicluster.samples())]))

    def testPatternBicluster4(self):

        # test case 4 - 11391x72 synthetic data with 238x19 pattern bicluster (additive and multiplicative type)
        data = SyntheticData((11391, 72))
        bicluster = data.implant_bicluster((238, 19), bicluster_type = BICLUSTER_PATTERN | TYPE_ADDITIVE | TYPE_MULTIPLICATIVE)
        self.assertTrue(isAdditiveAndMultiplicativeBicluster(data._array_[np.ix_(bicluster.genes(), bicluster.samples())]))

    def testPatternBicluster5(self):

        # test case 5 - 11391x72 synthetic data with manually specified pattern bicluster
        data = SyntheticData((11391, 72))
        implanted_genes = np.asarray([1642, 326, 3548, 3, 6336])
        implanted_samples = np.asarray([1, 6, 8, 3, 66])
        pi, ai, bi, qj, pj = 3.14159, np.random.uniform(size=(5, 1)), np.random.uniform(size=(5, 1)), np.random.uniform(size=(1, 5)), np.random.uniform(size=(1, 5))
        bicluster = data.implant_bicluster(
            size = (5, 5),
            pi = pi,
            ai = ai,
            bi = bi,
            qj = qj,
            pj = pj,
            genes = implanted_genes,
            samples = implanted_samples,
            bicluster_type = BICLUSTER_PATTERN | TYPE_ADDITIVE | TYPE_MULTIPLICATIVE
        )
        self.assertTrue(isAdditiveAndMultiplicativeBicluster(data._array_[np.ix_(implanted_genes, implanted_samples)]))
        self.assertTrue(sorted(bicluster.genes()) == sorted(implanted_genes))
        self.assertTrue(sorted(bicluster.samples()) == sorted(implanted_samples))

    def testTrendPreservingBicluster1(self):

        data = SyntheticData((300, 300))
        bicluster = data.implant_trend_preserving_bicluster((20, 20), 0)
        self.assertAlmostEqual(bicluster.ASR(), 1.0, places=2)

    def testTrendPreservingBicluster2(self):

        data = SyntheticData((300, 300))
        bicluster = data.implant_trend_preserving_bicluster((20, 20), 0, indices=([list(range(20)), list(range(20))]))
        self.assertAlmostEqual(bicluster.ASR(), 1.0, places=2)
        self.assertTrue(np.allclose(data.array()[np.ix_(list(range(20)), list(range(20)))], bicluster._values_))

if __name__ == "__main__":
    unittest.main()
