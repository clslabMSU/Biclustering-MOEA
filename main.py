"""
    PyClustering-Suite - A python wrapper and driver to facilitate the execution of biclustering algorithms.
    Biclustering algorithms included are not owned by me, and are downloaded from publicly available sources.
    Copyright (C) 2018 - Jeff Dale

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""


# Python libraries
from glob import glob
from os.path import sep
from pickle import dumps
from sys import stdout
from time import strftime, time
from typing import List

#import logging; logging.basicConfig(level=logging.DEBUG)
import numpy as np
import sqlite3

# Dependencies
from Bicluster import Bicluster
from BiclusterSet import BiclusterSet
from Dataset import Dataset

# Algorithms
# noinspection PyUnresolvedReferences
from algorithm.BicPAMS import BicPAMS
# noinspection PyUnresolvedReferences
from algorithm.ChengChurch import ChengChurch
# noinspection PyUnresolvedReferences
from algorithm.EvoBexpa import EvoBexpa
# noinspection PyUnresolvedReferences
from algorithm.FABIA import FABIA
# noinspection PyUnresolvedReferences
from algorithm.ISA import ISA
# noinspection PyUnresolvedReferences
from algorithm.NSGA_II import NSGA_II
# noinspection PyUnresolvedReferences
from algorithm.OPSM import OPSM
# noinspection PyUnresolvedReferences
from algorithm.PenalizedPlaid import PenalizedPlaid
# noinspection PyUnresolvedReferences
from algorithm.UniBic import UniBic

# Metrics
# noinspection PyUnresolvedReferences
from Metrics import VEt_fast as VEt
# noinspection PyUnresolvedReferences
from Metrics import SCS_fast as SCS
# noinspection PyUnresolvedReferences
from Metrics import ASR_fast as ASR
# noinspection PyUnresolvedReferences
from Metrics import TPC_fast as TPC
# noinspection PyUnresolvedReferences
from Metrics import MSR


print("""
******************************************************************
*                       PyClustering-Suite                       *
******************************************************************

""")


# TODO - windows support
def get_synthetic_datasets(path: str = "datasets/") -> List[Dataset]:
    files = [f for f in glob(path + "*.txt") if "_hiddenBics" not in f]
    datasets = []
    itr = 0
    for dataset in files:
        itr += 1
        stdout.write("\rGathering dataset %d/%d (%.2f%%)" % (itr, len(files), (100*float(itr)/len(files)))); stdout.flush()
        dataset_name = dataset.split("/")[-1].split(".")[0]
        with open(dataset, "r") as f:
            try:
                lines = f.readlines()
                matrix = np.asarray(list(map(lambda x : x[:-1].split("\t")[1:], lines[1:]))).astype(np.float32)
                gene_labels = list(map(lambda x:x.split("\t")[0], lines))[1:]
                sample_labels = lines[0].replace("\n", "").split("\t")[1:]
            except ValueError:
                print("Unable to parse file: " + dataset)
                continue
        with open(path + dataset_name + "_hiddenBics.txt", "r") as f:
            lines = list(filter(lambda x:"Bicluster" in x, f.readlines()))
            raw_bics = list(map(lambda x: list(map(lambda y: list(map(int, y.split(", "))), x[11:-3].split("], ["))), lines))
        known_bics = BiclusterSet([Bicluster(a[0], a[1], matrix[np.ix_(a[0], a[1])]) for a in raw_bics])
        datasets.append(Dataset(dataset_name, dataset, gene_labels, sample_labels, matrix, known_bics))

    stdout.write("\r"); stdout.flush()
    return datasets


# TODO - windows support
def get_plant_dataset(path: str = "PlantData/100% Present, Differential.txt"):
    with open(path, "r") as f:
        lines = f.readlines()
        matrix = np.asarray(list(map(lambda x: x[:-1].split("\t")[1:], lines[1:]))).astype(np.float32)
        gene_labels = list(map(lambda x: x.split("\t")[0].replace("\n", ""), lines))[1:]
        sample_labels = lines[0].replace("\n", "").split("\t")[1:]
    return [Dataset(path.split("/")[-1], path, gene_labels, sample_labels, matrix, None)]


def get_fabia_datasets(path: str = sep.join(["datasets", "fabia", ""])) -> (List[Dataset], List[Dataset]):

    datasets_noisy = []
    datasets_noise_free = []

    files = glob(path + "exp_*_X.txt")
    file_format = "_".join(files[0].split("_")[:-2]) + "_%d_%s.txt"
    for file in files:
        number = int(file.split("_")[-2])
        with open(file_format % (number, "L")) as f:
            # datasets are 1-indexed
            rows = list(map(lambda row: list(map(lambda x: int(x)-1, row.split("\t"))), f.readlines()))
        with open(file_format % (number, "Z")) as f:
            # datasets are 1-indexed
            cols = list(map(lambda row: list(map(lambda x: int(x)-1, row.split("\t"))), f.readlines()))

        noisy_matrix = np.loadtxt(file_format % (number, "X"), delimiter="\t", skiprows=1, dtype=np.float32)[:, 1:]
        noise_free_matrix = np.loadtxt(file_format % (number, "Y"), delimiter="\t", skiprows=1, dtype=np.float32)[:, 1:]

        # !!!!!!!!!!! TPC is here, do what you want with it (e.g. print, save, output to spreadsheet) !!!!!!!!!!!!!!!!!!
        tpc_noisy = TPC(noisy_matrix)
        tpc_noise_free = TPC(noise_free_matrix)

        known_bics_noisy = BiclusterSet()
        known_bics_noise_free = BiclusterSet()
        for row, col in zip(rows, cols):
            known_bics_noisy.append(Bicluster(row, col, noisy_matrix[np.ix_(row, col)]))
            known_bics_noise_free.append(Bicluster(row, col, noise_free_matrix[np.ix_(row, col)]))

        dataset_noisy = Dataset(file_format % (number, "X") + " (Noisy)", file_format % (number, "X"),
                                list(range(noisy_matrix.shape[0])),
                                list(range(noisy_matrix.shape[1])),
                                noisy_matrix,
                                known_bics_noisy)
        dataset_noise_free = Dataset(file_format % (number, "Y") + " (Noise Free)", file_format % (number, "Y"),
                                list(range(noise_free_matrix.shape[0])),
                                list(range(noise_free_matrix.shape[1])),
                                noise_free_matrix,
                                known_bics_noise_free)

        datasets_noisy.append(dataset_noisy)
        datasets_noise_free.append(dataset_noise_free)

    return datasets_noisy, datasets_noise_free


def start_at(datasets: List[Dataset], dataset: str) -> List[Dataset]:
    return datasets[list(map(lambda x:x.name, datasets)).index(dataset):]


def ASR_MIN(bic: np.ndarray) -> float:
    return 1 - abs(ASR(bic))

def TPC_MIN(bic: np.ndarray) -> float:
    return 1 - TPC(bic)


if __name__ == "__main__":

    Bicluster.LAZY = True
    tag = None

    execution_plan = {

        # list of tuples of the form (algorithm_instance, params_dict, num_reps)
        "algorithms": [
            #(BicPAMS, {}, 3),
            (ChengChurch, {"n_clusters": 20, "max_msr": 0.1}, 3),
            #(FABIA, {"num_bics": 10, "alpha": 0.01, "cycles": 500}, 3),
            #(ISA, {}, 3),
            #(OPSM, {"num_passed_models": 10}, 3),
            #(PenalizedPlaid, {"num_biclusters": 10, "method": "GPE", "sample": 1000, "burnin": 1000}, 3),
            #(QUBIC, {}, 1),
            #(UniBic, {}, 3),

            # NOTE FOR NSGA-II: ASR need to be MAXIMIZED IN ABSOLUTE VALUE, so run it through a lambda first to minimize -abs(ASR).
            #(NSGA_II, {
            #    "objective_functions": [MSR],
            #    "pop_size": 100,
            #    "max_generations": 1000,
            #    "crossover_rate": 0.85,
            #    "mutation_rate": 0.01,
            #    "generator_p_gene": 0.02,
            #    "generator_p_sample": 0.2,
            #    "replicates": 5,
            #    "hill_climber": True
            #}, 3),

            #(EvoBexpa, {}, 3)

        ],

        # list of Dataset objects to run algorithms on
        #"datasets": get_synthetic_datasets(),
        "datasets": [item for sublist in get_fabia_datasets() for item in sublist],
        #"datasets": get_plant_dataset()
        #"datasets": get_synthetic_datasets("synthetic/generated/narrow_trend_preserving/") +
        #            get_synthetic_datasets("synthetic/generated/square_trend_preserving/") +
        #            get_synthetic_datasets(),

        # whether to use lazy loading on internal validation measures, True is faster but may require extra computation later
        "lazy": True

    }

    results = {}
    num_parsed, num_total = 0.0, len(execution_plan["datasets"])
    Bicluster.LAZY = execution_plan["lazy"]

    # prepare database
    conn = sqlite3.connect("db/Pyclustering-DB.sqlite")
    c = conn.cursor()

    for dataset in execution_plan["datasets"]:

        assert(type(dataset) == Dataset)

        # TODO - REMOVE THIS ONCE ALGORITHM IS DESIGNED
        NSGA_II.GROUND_TRUTH = dataset.known_bics

        print("[ %.2f%% ] Current dataset: %s" % (100 * num_parsed / num_total, dataset.name))
        results[dataset.name] = {}
        num_parsed += 1

        for algorithm in execution_plan["algorithms"]:

            algo_cls, algo_params, algo_reps = algorithm
            algo_name = algo_cls.__name__

            results[dataset.name][algo_name] = []

            for rep in range(algo_reps):

                print("\t[%s] Executing algorithm: %s (Rep %d)" % (strftime("%Y.%m.%d %H:%M:%S"), algo_name, rep + 1))

                try:
                    # noinspection PyTypeChecker
                    results[dataset.name][algo_name].append(algo_cls.cluster(dataset, **algo_params))
                except KeyboardInterrupt:
                    print("Execution Interrupted.")
                    quit(1)

                # print recovery and relevance if they exist
                if dataset.known_bics is not None:
                    recovery = results[dataset.name][algo_name][-1].recovery(dataset.known_bics)
                    symmetric_recovery = results[dataset.name][algo_name][-1].symmetric_recovery(dataset.known_bics)
                    relevance = results[dataset.name][algo_name][-1].relevance(dataset.known_bics)
                    symmetric_relevance = results[dataset.name][algo_name][-1].symmetric_relevance(dataset.known_bics)
                    print("\t\tRecovery: " + str(recovery))
                    print("\t\tRelevance: " + str(relevance))
                    print("\t\tSymmetric Recovery: " + str(symmetric_recovery))
                    print("\t\tSymmetric Relevance: " + str(symmetric_relevance))
                else:
                    recovery, relevance = None, None
                    symmetric_recovery, symmetric_relevance = None, None

                # convert bicluster indices to gene/sample labels
                labels = [dataset.indices_to_labels(bic.genes(), bic.samples()) for bic in results[dataset.name][algo_name][-1]]

                # insert results into database
                c.execute("INSERT INTO log VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (
                    time(),
                    algo_name,
                    str(algo_params),
                    dataset.name,
                    dataset.md5,
                    "$".join(["|".join(list(map(lambda x: ",".join(x), bic))) for bic in labels]),
                    len(labels),
                    ",".join([str(len(x[0])) for x in labels]),
                    ",".join([str(len(x[1])) for x in labels]),
                    memoryview(dumps(results[dataset.name][algo_name][-1])),
                    symmetric_relevance,
                    symmetric_recovery,
                    tag if tag is not None else "null",
                    relevance,
                    recovery
                ))

                # commit to database
                conn.commit()

    conn.close()
