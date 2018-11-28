from os import name, system, remove
from sys import stdout

import numpy as np

from Bicluster import Bicluster
from BiclusterSet import BiclusterSet
from Dataset import Dataset


# TODO - refactor to work with unmodified UniBic binary (change out.blocks filename)
class UniBic:

    COMMAND = "./{folder}unibic -i \"{filename}\""

    def __init__(self):
        pass

    @staticmethod
    def cluster(dataset: Dataset, folder="algorithm/_UniBic/") -> BiclusterSet:

        stdout.write("\r\t\tRunning UniBic..."); stdout.flush()
        system(UniBic.COMMAND.format(folder=folder, filename=dataset.filename) + " > /dev/null" if name != "nt" else "")
        bics = []

        with open("out.blocks", "r") as f:
            lines = f.readlines()
            num_bics = len(list(filter(lambda x:x[:2] == "BC", lines)))
            bic = 0
            for i in range(len(lines)):
                if lines[i][:2] == "BC":
                    bic += 1
                    stdout.write("\r\t\tComputing bicluster %d/%d (%.2f%%)" % (bic, num_bics, 100 * float(bic) / num_bics)); stdout.flush()
                    #genes = list(map(lambda y:int(y[1:]), list(filter(lambda x:x!="", lines[i+1][:-1].split(": ")[1].split(" ")))))
                    #samples = list(map(lambda y:int(y[1:]), list(filter(lambda x:x!="", lines[i+2][:-1].split(": ")[1].split(" ")))))
                    # TODO - this is really slow, use a lookup table instead of index method
                    genes = list(map(lambda y: list(dataset.gene_labels).index(y), list(filter(lambda x:x!="", lines[i+1].replace("\n", "").split(": ")[1].split(" ")))))
                    samples = list(map(lambda y: list(dataset.sample_labels).index(y), list(filter(lambda x:x!="", lines[i+2].replace("\n", "").split(": ")[1].split(" ")))))
                    bics.append(Bicluster(genes, samples, dataset.matrix[np.ix_(genes, samples)]))

        stdout.write("\r"); stdout.flush()

        # remove excess files
        remove(dataset.filename + ".chars")
        remove("out.blocks")

        return BiclusterSet(bics)
