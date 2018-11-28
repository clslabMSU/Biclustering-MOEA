from functools import reduce
from statistics import mean
from typing import Tuple, Union
from warnings import warn

from Bicluster import Bicluster


# TODO - recovery and relevance
class BiclusterSet(list):

    def __init__(self, list_: list = None):
        if list_ is None:
            super().__init__()
        else:
            super().__init__(list_)


    def recovery(self, ground_truth: "BiclusterSet") -> float:
        #warn("Using standard recovery, you probably want symmetric recovery.")
        return BiclusterSet.match_score(ground_truth, self)


    def relevance(self, ground_truth: "BiclusterSet") -> float:
        #warn("Using standard relevance, you probably want symmetric relevance.")
        return BiclusterSet.match_score(self, ground_truth)

    def symmetric_recovery(self, ground_truth: "BiclusterSet") -> float:
        gene_rec, sample_rec = BiclusterSet.match_score(ground_truth, self, genes=True, samples=True)
        return 0.5*(gene_rec + sample_rec)

    def symmetric_relevance(self, ground_truth: "BiclusterSet") -> float:
        gene_rel, sample_rel = BiclusterSet.match_score(self, ground_truth, genes=True, samples=True)
        return 0.5*(gene_rel + sample_rel)

    def IR(self, *args) -> set:
        args = set(args)
        return set(filter(lambda x: args.issubset(x.genes()), self))

    def IC(self, *args) -> set:
        args = set(args)
        return set(filter(lambda x: args.issubset(x.samples()), self))


    def uniqueness(self, bicluster: Bicluster, row_index: int, col_index: int) -> float:
        """
        Quantifies the uniqueness of the information in the given bicluster with respect to the given row and column.

        Reference:
        Bozdağ, D., Parvin, J., & Catalyurek, U. (2009). A biclustering method to discover co-regulated genes using diverse gene expression datasets.
        Bioinformatics and Computational Biology, 151-163.

        :param bicluster: Bicluster object
        :param row_index: the row index to use
        :param col_index: the column index to use
        :return: Float between 0 and 1
        """
        total = 0.
        for x_i in bicluster.genes():

            # skip provided row_index
            if x_i == row_index: continue

            # compute IR(r, x_i)
            IR_r_xi = self.IR(x_i, row_index)

            for y_k in bicluster.samples():

                # compute IC(c, y_k)
                IC_c_yk = self.IC(y_k, col_index)
                total += 1.0/len(IR_r_xi.intersection(IC_c_yk))

        return total / ((len(bicluster.genes())-1)*len(bicluster.samples()))


    def overlap(self, row_index: int , col_index: int) -> float:
        """
        Calculate the overlap score

        Reference:
        Bozdağ, D., Parvin, J., & Catalyurek, U. (2009). A biclustering method to discover co-regulated genes using diverse gene expression datasets.
        Bioinformatics and Computational Biology, 151-163.

        :param row_index: the row index to use
        :param col_index: the column index to use
        :return: Float
        """

        IRrnICc = self.IR(row_index).intersection(self.IC(col_index))
        return sum(list(map(lambda bic: self.uniqueness(bic, row_index, col_index), IRrnICc)))


    def correlation_score(self, row_index: int) -> float:
        """
        Calculate the correlation score

        Reference:
        Bozdağ, D., Parvin, J., & Catalyurek, U. (2009). A biclustering method to discover co-regulated genes using diverse gene expression datasets.
        Bioinformatics and Computational Biology, 151-163.

        :param row_index: the row index to use
        :return: Float
        """

        all_samples = reduce(lambda x,y: x.union(set(y.samples())), self, set())
        return sum(self.overlap(row_index, c) for c in all_samples)


    @staticmethod
    def match_score(B1: "BiclusterSet", B2: "BiclusterSet", genes: bool = True, samples: bool = False) -> Union[float, Tuple[float, float]]:
        """
        Return the match score between two bicluster sets. If genes is True, match score will be calculated on the gene dimension.
        If samples is True, match score will be calculated on the sample dimension. If both are true, a tuple of (gene_ms, sample_ms)
        will be returned. Default behavior is as in the literature that proposed it.

        :param B1: first bicluster set
        :param B2: second bicluster set
        :param genes: whether to calculate gene match score
        :param samples: whether to calculate sample match score
        :return: desired match score(s)
        """

        if not (genes or samples):
            warn("Useless call to match_score function asking for no genes or samples.")

        ms_genes = 0.0
        ms_samples = 0.0

        for i in B1:
            if genes:
                ms_genes += max([float(len(list(set(i.genes()).intersection(set(j.genes()))))) /
                                 float(len(list(set(i.genes()).union(set(j.genes())))))
                                 for j in B2])
            if samples:
                ms_samples += max([float(len(list(set(i.samples()).intersection(set(j.samples()))))) /
                                   float(len(list(set(i.samples()).union(set(j.samples())))))
                                   for j in B2])

        if genes and not samples:
            return ms_genes / len(B1)
        if samples and not genes:
            return ms_samples / len(B1)

        return ms_genes / len(B1), ms_samples / len(B1)
