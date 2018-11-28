from Bicluster import Bicluster
from BiclusterSet import BiclusterSet
from Dataset import Dataset

import numpy as np
from os import devnull
import sys
import warnings

# hide output from this import
sys.stdout = open(devnull, "w"); import rpy2.robjects as robjects; sys.stdout = sys.__stdout__

import rpy2.robjects.numpy2ri; rpy2.robjects.numpy2ri.activate()
from rpy2.robjects.packages import importr


# TODO - dynamic output for reporting progress
class FABIA:

    def __init__(self):
        pass


    @staticmethod
    def cluster(dataset: Dataset, num_bics: int = 20, alpha: float = 0.01, cycles: int = 500) -> BiclusterSet:

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
            robjects.r("suppressMessages(library(fabia))")

            # perform FABIA biclustering
            robjects.r("FABIA.result = extractBic(fabia(data.matrix(input_data[-1, -1]), %d, %f, %d))" % (num_bics, alpha, cycles))

            # parse FABIA biclusters
            robjects.r("""
                biclusters = list()
                for (i in 1:dim(FABIA.result$bic)[[1]]) {
                    genes = as.numeric(unlist(lapply(FABIA.result$bic[i,]$bixn, FUN=function(x) return(substring(x, 2)))))
                    samples = as.numeric(unlist(lapply(FABIA.result$bic[i,]$biypn, FUN=function(x) return(substring(x, 2)))))
                    biclusters = c(biclusters, list(list(genes=genes, samples=samples)))
                }
            """)

            # reset stdout to default
            sys.stdout = sys.__stdout__

        # extract biclusters from R environment
        biclusters = BiclusterSet()
        for i in range(1, len(robjects.r("biclusters")) + 1):
            genes, samples = list(map(lambda x: list(map(int, x)), list(robjects.r("biclusters[[%d]]" % i))))
            biclusters.append(Bicluster(genes, samples, dataset.matrix[np.ix_(genes, samples)]))

        return biclusters



if __name__ == "__main__":

    install_command = "\"algorithm/_FABIA/fabia_2.24.0.tar.gz\", source=TRUE"
    base = importr('base')

    # evaluate locally a remote R script
    base.source("https://www.bioconductor.org/biocLite.R")
    robjects.r("biocLite(\"devtools\")")
    robjects.r("biocLite(%s)" % install_command)