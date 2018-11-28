from os import system
from sys import stdout

import numpy as np

from Bicluster import Bicluster
from BiclusterSet import BiclusterSet
from Dataset import Dataset


class PenalizedPlaid:

    COMMAND = "java -jar {folder}PenalizedPlaid.jar \"{folder}results/\" \"{folder}input.txt\" " + \
              "nbrgenes={num_genes} nbrconditions={num_conds} nbrofbiclusters={num_bic} {method} sample={sample} burnin={burnin} " + \
              ">> {folder}PenalizedPlaid.log"

    def __init__(self, input_file: str = None, matrix: np.ndarray = None, penalizedPlaidFolder: str = "algorithm/_PenalizedPlaid/"):
        self.filename = input_file
        self.folder = penalizedPlaidFolder
        if matrix is not None:
            self.needsConversion = False
            self.expression_data = matrix
            self.num_genes, self.num_samples = matrix.shape
            self.convert_file(readFirst=False, matrix=matrix)
        else:
            self.num_genes = None
            self.num_samples = None
            self.expression_data = None
            self.needsConversion = True


    def convert_file(self, readFirst: bool = True, matrix: np.ndarray = None):
        if readFirst:
            with open(self.filename, "r") as f:
                matrix = list(map(lambda x:x[:-1].split("\t")[1:], f.readlines()[1:]))
                self.expression_data = np.asarray(matrix).astype(float)
                self.num_genes = len(matrix)
                self.num_samples = len(matrix[0])
        if matrix is not None:
            with open(self.folder + "input.txt", "w") as f:
                f.write("\n".join(list(map(lambda x:"\t".join(list(map(str, x))), matrix))))
        else:
            raise ValueError("No matrix was provided for conversion.")


    def bicluster(self, num_biclusters: int = 10, method: str = "GPE", sample: int = 1000, burnin: int = 1000) -> BiclusterSet:

        stdout.write("\r\t\tRunning PenalizedPlaid..."); stdout.flush()

        if self.needsConversion:
            self.convert_file()
        prepared_command = PenalizedPlaid.COMMAND.format(
            folder = self.folder,
            num_genes = str(self.num_genes),
            num_conds = str(self.num_samples),
            num_bic = str(num_biclusters),
            method = method,
            sample = str(sample),
            burnin = str(burnin)
        )
        #print("Executing command:\n\t{command}".format(command=prepared_command))
        system(prepared_command)
        rho_file = "{folder}results/RhoEstimate.{method}.K{bics}.txt".format(folder=self.folder, method=method, bics=str(num_biclusters))
        kappa_file = "{folder}results/KappaEstimate.{method}.K{bics}.txt".format(folder=self.folder, method=method, bics=str(num_biclusters))
        return self.parse_results(rho_file, kappa_file)


    def parse_results(self, rho_file: str, kappa_file: str) -> BiclusterSet:

        with open(rho_file, "r") as f:
            genes_by_bicluster = np.asarray(list(map(lambda x:list(map(int, x[:-2].split(" "))), f.readlines())))
        with open(kappa_file, "r") as f:
            samples_by_bicluster = np.asarray(list(map(lambda x: list(map(int, x[:-2].split(" "))), f.readlines())))

        biclusters = []
        for i in range(genes_by_bicluster.shape[1]):
            stdout.write("\r\t\tComputing bicluster %d/%d (%.2f%%)" % (i+1, genes_by_bicluster.shape[1], 100 * float(i + 1) / genes_by_bicluster.shape[1])); stdout.flush()
            genes = np.where(genes_by_bicluster[:, i])[0].tolist()
            samples = np.where(samples_by_bicluster[:, i])[0].tolist()
            biclusters.append(Bicluster(genes, samples, self.expression_data[np.ix_(genes, samples)]))

        stdout.write("\r"); stdout.flush()

        return BiclusterSet(biclusters)


    @staticmethod
    def cluster(dataset: Dataset, **kwargs) -> BiclusterSet:
        self = PenalizedPlaid(matrix=dataset.matrix)
        return self.bicluster(**kwargs)


if __name__ == "__main__":
    PP = PenalizedPlaid(input_file="../datasets/narrow_100_10_data1.txt", penalizedPlaidFolder="_PenalizedPlaid/")
    bics = PP.bicluster()
    # noinspection PyTypeChecker
    ground_truth = BiclusterSet([
        Bicluster(list(range(0, 100)), list(range(0, 10)), None),
        Bicluster(list(range(100, 200)), list(range(10, 20)), None),
        Bicluster(list(range(200, 300)), list(range(20, 30)), None)
    ])
    print("Recovery score:\t" + str(bics.recovery(ground_truth)))
    print("Relevance score:\t" + str(bics.relevance(ground_truth)))
    for bic in bics:
        print(bic)