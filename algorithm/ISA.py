from Bicluster import Bicluster
from BiclusterSet import BiclusterSet
from Dataset import Dataset

import numpy as np
from os import devnull
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri; rpy2.robjects.numpy2ri.activate()
from rpy2.robjects.packages import importr
import sys
import warnings


class ISA:

    def __init__(self):
        pass


    @staticmethod
    def cluster(dataset: Dataset, num_bics: int = 20, alpha: float = 0.01, cycles: int = 500) -> BiclusterSet:

        sys.stdout.write("\t\tRunning ISA..."); sys.stdout.flush()

        # TODO - don't re-read files directly into R, use rpy2 to transfer existing numpy array
        with warnings.catch_warnings():

            warnings.simplefilter("ignore")

            # redirect stdout to cross-platform /dev/null to hide R output
            sys.stdout = open(devnull, "w")

            # re-read dataset into R environment
            robjects.r("input_data = read.table(\"%s\", header=FALSE, sep=\"\t\")" % dataset.filename)
            robjects.r("colnames(input_data) = as.character(unlist(input_data[1, ]))")
            robjects.r("rownames(input_data) = input_data[, 1]")

            # load FABIA library
            robjects.r("suppressMessages(library(isa2))")

            # perform FABIA biclustering
            robjects.r("isa.result = isa(data.matrix(input_data[-1, -1]))")

            # parse FABIA biclusters
            robjects.r("bicluster.rows = apply(isa.result$rows, MARGIN=2, function(b) which(b != 0, arr.ind=TRUE))")
            robjects.r("bicluster.cols = apply(isa.result$columns, MARGIN=2, function(b) which(b != 0, arr.ind=TRUE))")
            robjects.r("""
                biclusters = list()
                for (i in 1:length(bicluster.rows)) {
                    biclusters = c(biclusters, list(list(genes=bicluster.rows[[i]], samples=bicluster.cols[[i]])))
                }
            """)

            # reset stdout to default
            sys.stdout = sys.__stdout__

        # extract biclusters from R environment
        biclusters = BiclusterSet()
        for i in range(1, len(robjects.r("biclusters")) + 1):
            sys.stdout.write("\r\t\tComputing bicluster %d/%d (%.2f%%)" % (i, len(robjects.r("biclusters")), 100 * float(i) / len(robjects.r("biclusters")))); sys.stdout.flush()
            genes, samples = list(map(lambda x: list(map(lambda y: y-1, x)), list(robjects.r("biclusters[[%d]]" % i))))
            biclusters.append(Bicluster(genes, samples, dataset.matrix[np.ix_(genes, samples)]))
        sys.stdout.write("\r"); sys.stdout.flush()

        return biclusters



    @staticmethod
    def install(install_command: str = "algorithm/_isa2_0.3.5.tar.gz, source=TRUE") -> None:

        robjects.r("install.packages(\"%s\")" % install_command)