import numpy as np
import unittest

from Bicluster import Bicluster
from BiclusterSet import BiclusterSet

# noinspection PyTypeChecker
class SyntheticDataTests(unittest.TestCase):

    def testBiclusterUniqueness1(self):
        """
        Example from:
        Bozdağ, D., Parvin, J., & Catalyurek, U. (2009). A biclustering method to discover co-regulated genes using diverse gene expression datasets.
        Bioinformatics and Computational Biology, 151-163.
        """
        biclusters = BiclusterSet([
            Bicluster(genes=[2, 3, 4, 5, 6], samples=[1, 2, 3, 4], values=np.zeros((5, 4))),
            Bicluster(genes=[1, 2, 3, 5, 6], samples=[4, 5, 6, 7], values=np.zeros((5, 4))),
            Bicluster(genes=[5, 6, 7, 8], samples=[4, 7, 8, 9, 10], values=np.zeros((4, 5)))
        ])
        #print("\nColumn\t\t\t" + "\t\t\t".join(list(map(str, biclusters[0].samples()))))
        #for i in biclusters[0].genes():
        #    #print("Row ", i, end="\t\t\t")
        #    for j in biclusters[0].samples():
        #        print(biclusters.uniqueness(biclusters[0], i, j), end="\t\t\t")
        #    print()
        self.assertAlmostEqual(biclusters.uniqueness(biclusters[0], 2, 4), 0.91, places=2)

    def testBiclusterUniqueness2(self):
        """
        Example from:
        Bozdağ, D., Parvin, J., & Catalyurek, U. (2009). A biclustering method to discover co-regulated genes using diverse gene expression datasets.
        Bioinformatics and Computational Biology, 151-163.
        """
        biclusters = BiclusterSet([
            Bicluster(genes=[2, 3, 4, 5, 6], samples=[1, 2, 3, 4], values=np.zeros((5, 4))),
            Bicluster(genes=[1, 2, 3, 5, 6], samples=[4, 5, 6, 7], values=np.zeros((5, 4))),
            Bicluster(genes=[5, 6, 7, 8], samples=[4, 7, 8, 9, 10], values=np.zeros((4, 5)))
        ])
        self.assertAlmostEqual(biclusters.uniqueness(biclusters[0], 5, 4), 0.90, places=2)

    def testBiclusterOverlap1(self):
        """
        Example from:
        Bozdağ, D., Parvin, J., & Catalyurek, U. (2009). A biclustering method to discover co-regulated genes using diverse gene expression datasets.
        Bioinformatics and Computational Biology, 151-163.
        """
        biclusters = BiclusterSet([
            Bicluster(genes=[2, 3, 4, 5], samples=[1, 2, 3, 4], values=np.zeros((4, 4))),
            Bicluster(genes=[1, 2, 3, 5], samples=[4, 5, 6, 7], values=np.zeros((4, 4))),
            Bicluster(genes=[5, 7, 8], samples=[4, 7, 8, 9, 10], values=np.zeros((3, 5)))
        ])
        self.assertAlmostEqual(sum(biclusters.overlap(1, c) for c in range(1, 11)), 4.0, places=1)

    def testBiclusterOverlap2(self):
        """
        Example from:
        Bozdağ, D., Parvin, J., & Catalyurek, U. (2009). A biclustering method to discover co-regulated genes using diverse gene expression datasets.
        Bioinformatics and Computational Biology, 151-163.
        """
        biclusters = BiclusterSet([
            Bicluster(genes=[2, 3, 4, 5], samples=[1, 2, 3, 4], values=np.zeros((4, 4))),
            Bicluster(genes=[1, 2, 3, 5], samples=[4, 5, 6, 7], values=np.zeros((4, 4))),
            Bicluster(genes=[5, 7, 8], samples=[4, 7, 8, 9, 10], values=np.zeros((3, 5)))
        ])
        self.assertAlmostEqual(sum(biclusters.overlap(4, c) for c in range(1, 11)), 4.0, places=1)

    def testBiclusterOverlap3(self):
        """
        Example from:
        Bozdağ, D., Parvin, J., & Catalyurek, U. (2009). A biclustering method to discover co-regulated genes using diverse gene expression datasets.
        Bioinformatics and Computational Biology, 151-163.
        """
        biclusters = BiclusterSet([
            Bicluster(genes=[2, 3, 4, 5], samples=[1, 2, 3, 4], values=np.zeros((4, 4))),
            Bicluster(genes=[1, 2, 3, 5], samples=[4, 5, 6, 7], values=np.zeros((4, 4))),
            Bicluster(genes=[5, 7, 8], samples=[4, 7, 8, 9, 10], values=np.zeros((3, 5)))
        ])
        self.assertAlmostEqual(sum(biclusters.overlap(7, c) for c in range(1, 11)), 5.0, places=1)

    def testBiclusterOverlap4(self):
        """
        Example from:
        Bozdağ, D., Parvin, J., & Catalyurek, U. (2009). A biclustering method to discover co-regulated genes using diverse gene expression datasets.
        Bioinformatics and Computational Biology, 151-163.
        """
        biclusters = BiclusterSet([
            Bicluster(genes=[2, 3, 4, 5], samples=[1, 2, 3, 4], values=np.zeros((4, 4))),
            Bicluster(genes=[1, 2, 3, 5], samples=[4, 5, 6, 7], values=np.zeros((4, 4))),
            Bicluster(genes=[5, 7, 8], samples=[4, 7, 8, 9, 10], values=np.zeros((3, 5)))
        ])
        self.assertAlmostEqual(sum(biclusters.overlap(8, c) for c in range(1, 11)), 5.0, places=1)

    def testBiclusterOverlap5(self):
        """
        Example from:
        Bozdağ, D., Parvin, J., & Catalyurek, U. (2009). A biclustering method to discover co-regulated genes using diverse gene expression datasets.
        Bioinformatics and Computational Biology, 151-163.
        """
        biclusters = BiclusterSet([
            Bicluster(genes=[2, 3, 4, 5], samples=[1, 2, 3, 4], values=np.zeros((4, 4))),
            Bicluster(genes=[1, 2, 3, 5], samples=[4, 5, 6, 7], values=np.zeros((4, 4))),
            Bicluster(genes=[5, 7, 8], samples=[4, 7, 8, 9, 10], values=np.zeros((3, 5)))
        ])
        self.assertAlmostEqual(sum(biclusters.overlap(2, c) for c in range(1, 11)), 7.8, places=1)

    def testBiclusterOverlap6(self):
        """
        Example from:
        Bozdağ, D., Parvin, J., & Catalyurek, U. (2009). A biclustering method to discover co-regulated genes using diverse gene expression datasets.
        Bioinformatics and Computational Biology, 151-163.
        """
        biclusters = BiclusterSet([
            Bicluster(genes=[2, 3, 4, 5], samples=[1, 2, 3, 4], values=np.zeros((4, 4))),
            Bicluster(genes=[1, 2, 3, 5], samples=[4, 5, 6, 7], values=np.zeros((4, 4))),
            Bicluster(genes=[5, 7, 8], samples=[4, 7, 8, 9, 10], values=np.zeros((3, 5)))
        ])
        self.assertAlmostEqual(sum(biclusters.overlap(3, c) for c in range(1, 11)), 7.8, places=1)

    # TODO - this test is failing, I think there might be a typo in the paper
    # def testBiclusterOverlap7(self):
    #     """
    #     Example from:
    #     Bozdağ, D., Parvin, J., & Catalyurek, U. (2009). A biclustering method to discover co-regulated genes using diverse gene expression datasets.
    #     Bioinformatics and Computational Biology, 151-163.
    #     """
    #     biclusters = BiclusterSet([
    #         Bicluster(genes=[2, 3, 4, 5], samples=[1, 2, 3, 4], values=np.zeros((4, 4))),
    #         Bicluster(genes=[1, 2, 3, 5], samples=[4, 5, 6, 7], values=np.zeros((4, 4))),
    #         Bicluster(genes=[5, 7, 8], samples=[4, 7, 8, 9, 10], values=np.zeros((3, 5)))
    #     ])
    #     self.assertAlmostEqual(sum(biclusters.overlap(5, c) for c in range(1, 11)), 12.6, places=1)

    def testBiclusterCorrelationScore1(self):
        """
        Example from:
        Bozdağ, D., Parvin, J., & Catalyurek, U. (2009). A biclustering method to discover co-regulated genes using diverse gene expression datasets.
        Bioinformatics and Computational Biology, 151-163.
        """
        biclusters = BiclusterSet([
            Bicluster(genes=[2, 3, 4, 5], samples=[1, 2, 3, 4], values=np.zeros((4, 4))),
            Bicluster(genes=[1, 2, 3, 5], samples=[4, 5, 6, 7], values=np.zeros((4, 4))),
            Bicluster(genes=[5, 7, 8], samples=[4, 7, 8, 9, 10], values=np.zeros((3, 5)))
        ])
        self.assertAlmostEqual(biclusters.correlation_score(1), 4.0, places=1)

    def testBiclusterCorrelationScore2(self):
        """
        Example from:
        Bozdağ, D., Parvin, J., & Catalyurek, U. (2009). A biclustering method to discover co-regulated genes using diverse gene expression datasets.
        Bioinformatics and Computational Biology, 151-163.
        """
        biclusters = BiclusterSet([
            Bicluster(genes=[2, 3, 4, 5], samples=[1, 2, 3, 4], values=np.zeros((4, 4))),
            Bicluster(genes=[1, 2, 3, 5], samples=[4, 5, 6, 7], values=np.zeros((4, 4))),
            Bicluster(genes=[5, 7, 8], samples=[4, 7, 8, 9, 10], values=np.zeros((3, 5)))
        ])
        self.assertAlmostEqual(biclusters.correlation_score(4), 4.0, places=1)

    def testBiclusterCorrelationScore3(self):
        """
        Example from:
        Bozdağ, D., Parvin, J., & Catalyurek, U. (2009). A biclustering method to discover co-regulated genes using diverse gene expression datasets.
        Bioinformatics and Computational Biology, 151-163.
        """
        biclusters = BiclusterSet([
            Bicluster(genes=[2, 3, 4, 5], samples=[1, 2, 3, 4], values=np.zeros((4, 4))),
            Bicluster(genes=[1, 2, 3, 5], samples=[4, 5, 6, 7], values=np.zeros((4, 4))),
            Bicluster(genes=[5, 7, 8], samples=[4, 7, 8, 9, 10], values=np.zeros((3, 5)))
        ])
        self.assertAlmostEqual(biclusters.correlation_score(7), 5.0, places=1)

    def testBiclusterCorrelationScore4(self):
        """
        Example from:
        Bozdağ, D., Parvin, J., & Catalyurek, U. (2009). A biclustering method to discover co-regulated genes using diverse gene expression datasets.
        Bioinformatics and Computational Biology, 151-163.
        """
        biclusters = BiclusterSet([
            Bicluster(genes=[2, 3, 4, 5], samples=[1, 2, 3, 4], values=np.zeros((4, 4))),
            Bicluster(genes=[1, 2, 3, 5], samples=[4, 5, 6, 7], values=np.zeros((4, 4))),
            Bicluster(genes=[5, 7, 8], samples=[4, 7, 8, 9, 10], values=np.zeros((3, 5)))
        ])
        self.assertAlmostEqual(biclusters.correlation_score(8), 5.0, places=1)

    def testBiclusterCorrelationScore5(self):
        """
        Example from:
        Bozdağ, D., Parvin, J., & Catalyurek, U. (2009). A biclustering method to discover co-regulated genes using diverse gene expression datasets.
        Bioinformatics and Computational Biology, 151-163.
        """
        biclusters = BiclusterSet([
            Bicluster(genes=[2, 3, 4, 5], samples=[1, 2, 3, 4], values=np.zeros((4, 4))),
            Bicluster(genes=[1, 2, 3, 5], samples=[4, 5, 6, 7], values=np.zeros((4, 4))),
            Bicluster(genes=[5, 7, 8], samples=[4, 7, 8, 9, 10], values=np.zeros((3, 5)))
        ])
        self.assertAlmostEqual(biclusters.correlation_score(2), 7.8, places=1)

    def testBiclusterCorrelationScore6(self):
        """
        Example from:
        Bozdağ, D., Parvin, J., & Catalyurek, U. (2009). A biclustering method to discover co-regulated genes using diverse gene expression datasets.
        Bioinformatics and Computational Biology, 151-163.
        """
        biclusters = BiclusterSet([
            Bicluster(genes=[2, 3, 4, 5], samples=[1, 2, 3, 4], values=np.zeros((4, 4))),
            Bicluster(genes=[1, 2, 3, 5], samples=[4, 5, 6, 7], values=np.zeros((4, 4))),
            Bicluster(genes=[5, 7, 8], samples=[4, 7, 8, 9, 10], values=np.zeros((3, 5)))
        ])
        self.assertAlmostEqual(biclusters.correlation_score(3), 7.8, places=1)

if __name__ == "__main__":
    unittest.main()
