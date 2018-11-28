import numpy as np
from warnings import warn

from Metrics import MSR, SMSR, VEt_fast as VEt, ASR_fast as ASR, SCS_fast as SCS, TPC_fast as TPC


class Bicluster:

    LAZY = False

    def __init__(self, genes: list, samples: list, values: np.ndarray):
        """
        Bicluster class to facilitate operations on biclusters.

        :param genes: list of gene indices in bicluster
        :param samples: list of sample indices in bicluster
        :param values: expression values in bicluster, typically sliced by matrix[np.ix_(genes, samples)]
        """

        assert(values is not None)

        # internal members
        self._genes_ = genes
        self._samples_ = samples
        self._values_ = values

        # bicluster quality measures
        self._MSR_ = MSR(self._values_) if not Bicluster.LAZY else None
        self._SMSR_ = SMSR(self._values_) if not Bicluster.LAZY else None
        #self._VE_ = None #VE(self._values_) if not Bicluster.LAZY else None
        self._VEt_ = VEt(self._values_) if not Bicluster.LAZY else None
        self._ASR_ = ASR(self._values_) if not Bicluster.LAZY else None
        self._SCS_ = SCS(self._values_) if not Bicluster.LAZY else None
        self._TPC_ = TPC(self._values_) if not Bicluster.LAZY else None

        #self._volume_ = bicluster_volume(self._values_) if not Bicluster.LAZY else None
        self.chromosome = None


    def genes(self):
        return self._genes_

    def samples(self):
        return self._samples_

    def values(self):
        return self._values_

    def MSR(self):
        if self._MSR_ is None:
            self._MSR_ = MSR(self._values_)
        return self._MSR_
    
    def SMSR(self):
        if self._SMSR_ is None:
            self._SMSR_ = SMSR(self._values_)
        return self._SMSR_
    
    def VE(self):
        warn("VE is not implemented in Cython yet, and no plans exist to do so.")
        return None
        #if self._VE_ is None:
        #    self._VE_ = VE(self._values_)
        #return self._VE_
    
    def VEt(self):
        if self._VEt_ is None:
            self._VEt_ = VEt(self._values_)
        return self._VEt_

    def ASR(self):
        if self._ASR_ is None:
            self._ASR_ = ASR(self._values_)
        return self._ASR_

    def SCS(self):
        # backward compatibility with previously pickled biclusters
        if not hasattr(self, "_SCS_"):
            self._SCS_ = None
        if self._SCS_ is None:
            self._SCS_ = SCS(self._values_)
        return self._SCS_

    def TPC(self):
        # backward compatibility with previously pickled biclusters
        if not hasattr(self, "_TPC_"):
            self._TPC_ = None
        if self._TPC_ is None:
            self._TPC_ = TPC(self._values_)
        return self._TPC_


    def __str__(self):
        ret  = "Bicluster of size (%d, %d):" % (len(self._genes_), len(self._samples_)) + "\n"
        ret += "\tGenes: " + ", ".join(list(map(str, sorted(self._genes_)))) + "\n"
        ret += "\tSamples: " + ", ".join(list(map(str, sorted(self._samples_)))) + "\n"
        ret += "\tMSR:\t" + str(self.MSR()) + "\n"
        ret += "\tSMSR:\t" + str(self.SMSR()) + "\n"
        ret += "\tVE: \t" + str(self.VE()) + "\n"
        ret += "\tVEt:\t" + str(self.VEt()) + "\n"
        ret += "\tASR:\t" + str(self.ASR()) + "\n"
        ret += "\tSCS:\t" + str(self.SCS()) + "\n"
        ret += "\tTPC:\t" + str(self.TPC()) + "\n"
        return ret


    @staticmethod
    def from_chromosome(chromosome: np.ndarray, dataset) -> "Bicluster":
        genes = np.where(chromosome[:dataset.matrix.shape[0]] == 1)[0]
        samples = np.where(chromosome[dataset.matrix.shape[0]:] == 1)[0]
        values = dataset.matrix[np.ix_(genes, samples)]
        bic = Bicluster(genes, samples, values)
        bic.chromosome = chromosome
        return bic

