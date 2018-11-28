from os import name, system
from sys import stdout

import numpy as np

from Bicluster import Bicluster
from BiclusterSet import BiclusterSet
from Dataset import Dataset


class OPSM:

    COMMAND = "java -jar {folder}OPSM.jar {folder}opsm_tmp.txt {rows} {cols} {folder}opsm_tmp.out {passed_models}"

    def __init__(self):
        pass

    @staticmethod
    def cluster(dataset: Dataset, num_passed_models: int = 10, folder="algorithm/_OPSM/") -> BiclusterSet:

        stdout.write("\r\t\tRunning OPSM...")

        np.savetxt(folder + "opsm_tmp.txt", dataset.matrix, delimiter="\t")
        rows, cols = dataset.matrix.shape
        system(OPSM.COMMAND.format(folder=folder, rows=str(rows), cols=str(cols), passed_models=str(num_passed_models)) + " > /dev/null" if name != "nt" else "")
        clusters = []

        with open("{folder}opsm_tmp.out".format(folder=folder), "r") as f:
            lines = list(map(lambda x:list(map(int, x[:-1].split(" "))) if x != "\n" else "", f.readlines()))
            for i in range(len(lines) // 3):
                stdout.write("\r\t\tComputing bicluster %d/%d (%.2f%%)" % (i+1, len(lines) // 3, 100 * float(i+1) / (len(lines) // 3))); stdout.flush()
                genes = list(filter(lambda x: x < len(dataset.gene_labels), lines[i*3]))
                samples = list(filter(lambda x: x < len(dataset.gene_labels), lines[i*3 + 1]))
                values = dataset.matrix[np.ix_(genes, samples)]
                clusters.append(Bicluster(genes, samples, values))

        stdout.write("\r"); stdout.flush()
        return BiclusterSet(clusters)