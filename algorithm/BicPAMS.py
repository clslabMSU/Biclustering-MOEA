from bs4 import BeautifulSoup
from os import name, system
from shutil import copyfile
from sys import stdout

import numpy as np

from Bicluster import Bicluster
from BiclusterSet import BiclusterSet
from Dataset import Dataset


class BicPAMS:

    COMMAND = "java -jar {folder}bicpams_4.0.3_console.jar --input=/{folder}/bicpams_tmp.txt --output=/{folder}/bicpams_tmp.out"

    def __init__(self):
        pass

    @staticmethod
    def cluster(dataset: Dataset, folder="algorithm/_BicPAMS/") -> BiclusterSet:
        copyfile(dataset.filename, folder + "bicpams_tmp.txt")

        stdout.write("\t\tRunning BicPAMS..."); stdout.flush()
        system(BicPAMS.COMMAND.format(folder=folder) + " > /dev/null" if name != "nt" else "")

        biclusters = BiclusterSet()

        with open("{folder}bicpams_tmp.out".format(folder=folder), "r") as f:
            soup = BeautifulSoup(f.read(), "lxml")
            genes_and_samples = list(map(lambda x:[eval(x.select("td")[1].contents[y][1:]) for y in [1, 4]], soup.select("table")[0].select("tr")))
            for i in range(len(genes_and_samples)):
                stdout.write("\r\t\tComputing bicluster %d/%d (%.2f%%)" % (i+1, len(genes_and_samples), 100*float(i+1) / len(genes_and_samples))); stdout.flush()
                biclusters.append(Bicluster(*genes_and_samples[i], dataset.matrix[np.ix_(genes_and_samples[i][0], genes_and_samples[i][1])]))
            stdout.write("\r"); stdout.flush()

        return BiclusterSet(biclusters)