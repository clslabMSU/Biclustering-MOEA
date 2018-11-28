from hashlib import md5
import numpy as np
from typing import List, Union

from BiclusterSet import BiclusterSet


class Dataset:

    def __init__(self, name: str, filename: str, gene_labels: Union[List[str], np.ndarray], sample_labels: Union[List[str], np.ndarray], matrix: np.ndarray, known_bics: BiclusterSet = None):
        self.name = name
        self.filename = filename
        self.gene_labels = np.array(gene_labels)
        self.sample_labels = np.array(sample_labels)
        self.matrix = matrix
        self.known_bics = known_bics
        try:
            self.md5 = md5(open(filename, 'rb').read()).hexdigest()
        except FileNotFoundError:
            self.md5 = None


    def indices_to_labels(self, gene_indices: Union[np.ndarray, list], sample_indices: Union[np.ndarray, list]) -> (list, list):
        """
        Cleaner way to convert a list of gene/sample indices to their corresponding labels.

        :param gene_indices: list of gene indices
        :param sample_indices: list of sample indices
        :return:
        """
        gene_labels = list(map(lambda x: self.gene_labels[x], gene_indices))
        sample_labels = list(map(lambda x: self.sample_labels[x], sample_indices))
        return gene_labels, sample_labels


    def write_unibic_format(self, folder: str = "generated/", ground_truth: bool = True) -> None:
        """
        Write dataset to tab-separated file in the same format as UniBic datasets.

        :param folder: folder to save dataset in
        :param ground_truth: whether ground truth is included and should be written
        :return: None
        """
        with open(folder + self.filename, "w") as f:
            out_str = "genes\t" + "\t".join(self.sample_labels) + "\n"
            for i in range(self.matrix.shape[0]):
                out_str += self.gene_labels[i] + "\t" + "\t".join(list(map(str, self.matrix[i, :]))) + "\n"
            f.write(out_str)

        if ground_truth:
            with open(folder + ".".join(self.filename.split(".")[:-1]) + "_hiddenBics.txt", "w") as f:
                out_str = "#" + str(len(self.known_bics)) + "\n"
                for bic in self.known_bics:
                    out_str += "Bicluster([%s], [%s])\n" % (", ".join(list(map(str, bic.genes()))), ", ".join(list(map(str, bic.samples()))))
                f.write(out_str)
