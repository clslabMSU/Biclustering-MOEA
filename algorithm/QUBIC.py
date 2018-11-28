from os import name, system, remove

import numpy as np

from Bicluster import Bicluster
from BiclusterSet import BiclusterSet
from Dataset import Dataset


# TODO - dynamic output for reporting progress
class QUBIC:

    COMMAND = "./{folder}qubic -i \"{filename}\""

    def __init__(self):
        pass

    @staticmethod
    def cluster(dataset: Dataset, folder="algorithm/_QUBIC/") -> BiclusterSet:
        system(QUBIC.COMMAND.format(folder=folder, filename=dataset.filename) + " > /dev/null" if name != "nt" else "")
        bics = []

        with open(dataset.filename + ".blocks", "r") as f:
            lines = f.readlines()
            for i in range(len(lines)):
                if lines[i][:2] == "BC":
                    genes = list(map(lambda y:int(y[1:]), list(filter(lambda x:x!="", lines[i+1][:-1].split(": ")[1].split(" ")))))
                    samples = list(map(lambda y:int(y[1:]), list(filter(lambda x:x!="", lines[i+2][:-1].split(": ")[1].split(" ")))))
                    bics.append(Bicluster(genes, samples, dataset.matrix[np.ix_(genes, samples)]))

        # remove excess files
        remove(dataset.filename + ".chars")
        remove(dataset.filename + ".blocks")
        remove(dataset.filename + ".rules")

        return BiclusterSet(bics)
