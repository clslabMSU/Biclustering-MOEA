from copy import copy
from functools import reduce
import matplotlib.pyplot as plt
from multiprocessing import Process
from multiprocessing import Manager
from random import Random
from sys import stdout
from typing import List

import numpy as np

from inspyred_numpy.ec import DiscreteBounder, Individual
from inspyred_numpy.ec.emo import NSGA2 as inspyred_NSGA2
from inspyred_numpy.ec.emo import Pareto
from inspyred_numpy.ec.terminators import generation_termination
from inspyred_numpy.ec.variators import bit_flip_mutation
from inspyred_numpy.ec.variators import crossover
from inspyred_numpy.benchmarks import Benchmark

from Bicluster import Bicluster
from BiclusterSet import BiclusterSet
from Dataset import Dataset


# TODO - work with vanilla Inspyred
class NSGA_II:

    DATASET = None
    GENERATIONS = None
    EVALUATIONS = None
    ARCHIVES = None
    FITNESS_FUNCTIONS = None
    PLOT = False

    # TODO - REMOVE THIS AFTER ALGORITHM IS DESIGNED
    GROUND_TRUTH = None
    RECOVERIES, RELEVANCES = None, None

    def __init__(self):
        pass


    @staticmethod
    def cluster(dataset: Dataset,
                objective_functions: List[callable] = None,
                pop_size: int = 50,
                max_generations: int = 100,
                crossover_rate: float = 0.8,
                mutation_rate: float = 0.05,
                num_crossover_points: int = 1,
                generator_p_gene: float = 0.5,
                generator_p_sample: float = 0.5,
                replicates: int = 1,
                hill_climber: bool = False
        ) -> BiclusterSet:

        if objective_functions is None:
            raise TypeError

        # get all parameters of function to pass to parallel cluster function
        kwargs = locals()

        NSGA_II.DATASET = dataset

        # prepare storage of results
        manager = Manager()
        result = manager.list()
        NSGA_II.GENERATIONS = manager.list(range(replicates))
        NSGA_II.EVALUATIONS = manager.list(range(replicates))
        NSGA_II.ARCHIVES = manager.list([None for _ in range(replicates)])
        NSGA_II.FITNESS_FUNCTIONS = objective_functions

        # TODO - REMOVE THIS WHEN ALGORITHM IS DESIGNED
        NSGA_II.RECOVERIES = manager.list(range(replicates))
        NSGA_II.RELEVANCES = manager.list(range(replicates))

        # run algorithm in parallel
        procs = []
        for rep in range(replicates):
            procs.append(Process(target=NSGA_II.cluster_parallel, args=(result,rep), kwargs=kwargs))
            procs[-1].start()

        [procs[i].join() for i in range(len(procs))]

        final_arc = reduce(lambda x, y: x + y, list(result))
        biclusters = NSGA_II.archive_to_bicluster_set(final_arc, dataset)


        #if len(objective_functions) == 2:
        #    NSGA_II.pareto_plot(ea, "Pareto Optimality of Final Archive")

        if hill_climber:
            rec, rel = biclusters.symmetric_recovery(dataset.known_bics), biclusters.symmetric_relevance(dataset.known_bics)
            biclusters = NSGA_II.hillclimber(biclusters)
            # with open("results_old.txt", "a") as f:
            #     f.write(",".join([dataset.name, str(rec), str(rel), str(biclusters.recovery(dataset.known_bics)), str(biclusters.relevance(dataset.known_bics))]) + "\n")
            # print("Wrote results to file.")
        stdout.write("\r")
        stdout.flush()

        return biclusters


    @staticmethod
    def cluster_parallel(result_list, replicate, **kwargs) -> None:

        problem = BiclusteringProblem(kwargs["dataset"], kwargs["objective_functions"])

        # TODO - make the random parameterized for replicability
        ea = inspyred_NSGA2(Random())
        ea.variator = [BiclusteringProblem.crossover,
                       bit_flip_mutation]
        ea.archiver = BiclusteringProblem.archiver
        ea.terminator = generation_termination
        ea.observer = NSGA_II.parallel_progress_observer
        ea.replicate = replicate

        # add field to EA so it can be accessed in progress_observer
        ea.problem = problem

        try:
            ea.evolve(
                generator = problem.generator,
                evaluator = problem.evaluator,
                pop_size = kwargs["pop_size"],
                maximize = problem.maximize,
                bounder = problem.bounder,
                max_generations = kwargs["max_generations"],
                crossover_rate = kwargs["crossover_rate"],
                num_crossover_points = kwargs["num_crossover_points"],
                mutation_rate = kwargs["mutation_rate"],
                generator_p_gene = kwargs["generator_p_gene"],
                generator_p_sample = kwargs["generator_p_sample"],
                dataset = kwargs["dataset"]
            )

        # handle CTRL+C more elegantly than filling the screen with error tracebacks
        except KeyboardInterrupt:
            print("Execution interrupted, process terminated.")
            quit(1)

        result_list.append(ea.archive)


    @staticmethod
    def hillclimber(biclusters: BiclusterSet, ratio: float = 1.25, bias: float = 0.01, max_size: (int, int) = (100, 50)) -> BiclusterSet:
        """
        Local hill climbing algorithm for increasing the size of biclusters.
        The fitness value of the returned set of biclusters is at most ratio*original_fitness_value + bias.

        :param biclusters: biclusters to improve upon
        :param ratio: percent of fitness to preserve
        :param bias: added scalar to ratio*original_fitness_vlaue for when fitness is extremely low
        :param max_size: maximum size to expand the bicluster
        :return: updated list of biclusters
        """

        new_biclusters = BiclusterSet()

        for bicluster in biclusters:

            # get original fitness values
            old_fitnesses = np.array([NSGA_II.FITNESS_FUNCTIONS[i](bicluster._values_) for i in range(len(NSGA_II.FITNESS_FUNCTIONS))])

            # as long as new fitnesses haven't gone below ratio*old_fitnesses, continue adding rows/columns
            while np.all(np.array([NSGA_II.FITNESS_FUNCTIONS[i](bicluster._values_) for i in range(len(NSGA_II.FITNESS_FUNCTIONS))]) <= ratio*old_fitnesses + bias):
                available_genes = list(filter(lambda x: x not in bicluster.genes(), range(NSGA_II.DATASET.matrix.shape[0])))
                available_samples = list(filter(lambda x: x not in bicluster.samples(), range(NSGA_II.DATASET.matrix.shape[1])))

                if len(available_genes)*len(available_samples) == 0:
                    #print("No available genes/samples to use, stopping expansion.")
                    break

                if bicluster._values_.shape[0] > max_size[0] and bicluster._values_.shape[1] > max_size[1]:
                    #print("Bicluster size too large, stopping expansion.")
                    break

                # TODO - incrementally calculate this
                # Greedily calculate the best gene and best sample to add to the bicluster
                best_added_gene = min(available_genes,
                    key=lambda x: np.linalg.norm(np.array([NSGA_II.FITNESS_FUNCTIONS[i](NSGA_II.DATASET.matrix[np.ix_(bicluster.genes() + [x], bicluster.samples())])
                        for i in range(len(NSGA_II.FITNESS_FUNCTIONS))]))
                )
                best_added_sample = min(available_samples,
                    key=lambda x: np.linalg.norm(np.array([NSGA_II.FITNESS_FUNCTIONS[i](NSGA_II.DATASET.matrix[np.ix_(bicluster.genes(), bicluster.samples() + [x])])
                        for i in range(len(NSGA_II.FITNESS_FUNCTIONS))]))
                )

                # calculate the fitness of the biclusters including the new gene and new sample (separately)
                gene_fitness = np.linalg.norm(np.array([NSGA_II.FITNESS_FUNCTIONS[i](NSGA_II.DATASET.matrix[np.ix_(bicluster.genes() + [best_added_gene], bicluster.samples())])
                                    for i in range(len(NSGA_II.FITNESS_FUNCTIONS))]))
                sample_fitness = np.linalg.norm(np.array([NSGA_II.FITNESS_FUNCTIONS[i](NSGA_II.DATASET.matrix[np.ix_(bicluster.genes(), bicluster.samples()+ [best_added_sample])])
                                    for i in range(len(NSGA_II.FITNESS_FUNCTIONS))]))

                # if adding the gene gives lower fitness than adding the sample, add the gene. Otherwise, add the sample.
                if gene_fitness <= sample_fitness and bicluster._values_.shape[0] < max_size[0]:
                    #print("Adding gene...")
                    new_genes = sorted(bicluster.genes() + [best_added_gene])
                    bicluster = Bicluster(new_genes, bicluster.samples(), NSGA_II.DATASET.matrix[np.ix_(new_genes, bicluster.samples())])
                elif sample_fitness <= gene_fitness and bicluster._values_.shape[1] < max_size[1]:
                    #print("Adding sample...")
                    new_samples = sorted(bicluster.samples() + [best_added_sample])
                    bicluster = Bicluster(bicluster.genes(), new_samples, NSGA_II.DATASET.matrix[np.ix_(bicluster.genes(), new_samples)])
                else:
                    break

            # append the optimized bicluster to the return list
            new_biclusters.append(bicluster)

        return new_biclusters


    @staticmethod
    def archive_to_bicluster_set(archive, dataset: Dataset) -> BiclusterSet:
        biclusters = BiclusterSet()
        for solution in archive:
            genes = np.where(solution.candidate[:dataset.matrix.shape[0]])[0]
            samples = np.where(solution.candidate[dataset.matrix.shape[0]:])[0]
            biclusters.append(Bicluster(list(genes), list(samples), dataset.matrix[np.ix_(genes, samples)]))
        return biclusters


    # noinspection PyUnusedLocal
    @staticmethod
    def progress_observer_plot(population, num_generations, num_evaluations, args):
        if len(args["_ec"].problem.objective_functions) == 2:
            NSGA_II.pareto_plot(args["_ec"], "Pareto Optimality of Solutions (Gen %d)" % num_generations)
        stdout.write("\r\t\tGeneration: %d\t\tEvaluations: %d" % (num_generations, num_evaluations))
        stdout.flush()


    # noinspection PyUnusedLocal
    @staticmethod
    def parallel_progress_observer(population, num_generations, num_evaluations, args):
        rep = args["_ec"].replicate
        NSGA_II.GENERATIONS[rep] = num_generations
        NSGA_II.EVALUATIONS[rep] = num_evaluations
        NSGA_II.ARCHIVES[rep] = args["_ec"].archive
        if None in list(NSGA_II.ARCHIVES):
            return
        fitnesses = [np.array([ind.fitness.values for ind in NSGA_II.ARCHIVES[r]]) for r in range(len(NSGA_II.GENERATIONS))]
        mins = [[min(f[:, i]) for i in range(len(NSGA_II.FITNESS_FUNCTIONS))] for f in fitnesses]

        # TODO - REMOVE THIS WHEN ALGORITHM IS DESIGNED
        all_archives = reduce(lambda x, y: x + y, list(NSGA_II.ARCHIVES))
        bics = NSGA_II.archive_to_bicluster_set(all_archives, args["_ec"].problem.dataset)
        recovery, relevance = bics.symmetric_recovery(NSGA_II.GROUND_TRUTH), bics.symmetric_relevance(NSGA_II.GROUND_TRUTH)

        stdout.write("\r\t\tGens: " + repr(list(NSGA_II.GENERATIONS)) + \
                     "\t\tEvals: " + repr(list(NSGA_II.EVALUATIONS)) + \
                     "\t\tMins: " + repr([[round(float(j), 6) for j in i] for i in mins])[1:-1] + \
                     "\t\t|Archives|: " + repr(list(map(len, NSGA_II.ARCHIVES))) + \
                     # TODO - REMOVE THIS WHEN ALGORITHM IS DESIGNED
                     "\t\tSRec/SRel: [%.3f, %.3f]" % (recovery, relevance)
        )
        stdout.flush()

        if NSGA_II.PLOT:
            fig, axarr = plt.subplots(2, 2)
            for i, j in zip([0, 0, 1, 1], [0, 1, 0, 1]):
                x = []
                y = []
                for f in NSGA_II.ARCHIVES[2*i + j]:
                    x.append(f.fitness[0])
                    y.append(f.fitness[1])
                axarr[i, j].scatter(x, y, color='b')
                axarr[i, j].set_title("Replicate %d, Gen %d" % (2*i + j, NSGA_II.GENERATIONS[2*i + j]))
                axarr[i, j].set_xlabel(NSGA_II.FITNESS_FUNCTIONS[0].__name__)
                axarr[i, j].set_ylabel(NSGA_II.FITNESS_FUNCTIONS[1].__name__)

            fig.subplots_adjust(hspace=0.7, wspace=0.5)
            plt.show()



    @staticmethod
    def pareto_plot(ec, title):
        x = []
        y = []
        for f in ec.archive:
            x.append(f.fitness[0])
            y.append(f.fitness[1])
        plt.scatter(x, y, color='b')
        plt.title(title)
        plt.xlabel(ec.problem.objective_functions[0].__name__)
        plt.ylabel(ec.problem.objective_functions[1].__name__)
        plt.show()


class BiclusteringProblem(Benchmark):

    MIN_GENES = 10
    MIN_SAMPLES = 10
    ARCHIVE_OVERLAP_THRESHOLD = 0.5

    def __init__(self, dataset: Dataset, objective_functions: List[callable]):
        """
        Definition of the biclustering problem for compatibility with Inspyred

        :param dataset: dataset to bicluster
        :param objective_functions: list of functions with signature f(np.ndarray) -> float
        """

        self.dataset = dataset

        # number of genes + number of samples
        self.dimensions = dataset.matrix.shape[0] + dataset.matrix.shape[1]

        # specify any list of objective functions
        self.objectives = len(objective_functions)
        self.objective_functions = objective_functions

        Benchmark.__init__(self, self.dimensions, self.objectives)

        # values are discrete and binary
        self.bounder = DiscreteBounder([0, 1])

        # we want to minimize the objective functions
        self.maximize = False


    def generator(self, random: Random, args: dict) -> np.ndarray:
        """
        Generate random solution.

        :param random: instance of random
        :param args: key "generator_p" is probability that a random bit will be one
        :return: list of binary values
        """
        args.setdefault("generator_p_gene", 0.5)
        args.setdefault("generator_p_sample", 0.5)
        genes = np.random.choice([1, 0], self.dataset.matrix.shape[0], replace=True, p=[args["generator_p_gene"], 1-args["generator_p_gene"]])
        samples = np.random.choice([1, 0], self.dataset.matrix.shape[1], replace=True, p=[args["generator_p_sample"], 1-args["generator_p_sample"]])
        return np.concatenate((genes, samples))


    def evaluator(self, candidates: List[np.ndarray], args: dict) -> List[Pareto]:
        """
        Evaluate list of candidate solutions.

        :param candidates: list of candidate solutions
        :param args: unused
        :return: list of pareto fitnesses
        """
        fitnesses = []
        for candidate in candidates:
            genes = np.where(candidate[:self.dataset.matrix.shape[0]] == 1)[0]
            samples = np.where(candidate[self.dataset.matrix.shape[0]:] == 1)[0]
            if len(genes) < BiclusteringProblem.MIN_GENES or len(samples) < BiclusteringProblem.MIN_SAMPLES:
                fitnesses.append(Pareto([np.inf for _ in self.objective_functions]))
            else:
                bicluster = self.dataset.matrix[np.ix_(genes, samples)]
                fitnesses.append(Pareto([f(bicluster) for f in self.objective_functions]))
        return fitnesses

    @staticmethod
    @crossover
    def crossover(random: Random, mom: List[np.ndarray], dad: List[np.ndarray], args: dict) -> List[np.ndarray]:

        dataset = args["dataset"]

        mom_genes, dad_genes = mom[:dataset.matrix.shape[0]], dad[:dataset.matrix.shape[0]]
        mom_samples, dad_samples = mom[dataset.matrix.shape[0]:], dad[dataset.matrix.shape[0]:]

        child1_genes, child2_genes = BiclusteringProblem.n_point_crossover(random, mom_genes, dad_genes, args)
        child1_samples, child2_samples = BiclusteringProblem.n_point_crossover(random, mom_samples, dad_samples, args)

        return [np.concatenate((child1_genes, child1_samples)), np.concatenate((child2_genes, child2_samples))]


    @staticmethod
    def n_point_crossover(random, mom, dad, args):
        """
        Copy of Inspyred's n-point crossover, without the "crossover" decorator.
        """
        crossover_rate = args.setdefault('crossover_rate', 1.0)
        num_crossover_points = args.setdefault('num_crossover_points', 1)
        children = []
        if random.random() < crossover_rate:
            num_cuts = min(len(mom) - 1, num_crossover_points)
            cut_points = random.sample(range(1, len(mom)), num_cuts)
            cut_points.sort()
            bro = copy(dad)
            sis = copy(mom)
            normal = True
            for i, (m, d) in enumerate(zip(mom, dad)):
                if i in cut_points:
                    normal = not normal
                if not normal:
                    bro[i] = m
                    sis[i] = d
            children.append(bro)
            children.append(sis)
        else:
            children.append(mom)
            children.append(dad)
        return children


    @staticmethod
    def archiver(random, population, archive, args) -> list:
        """
        Extension of Inspyred's archiver
        """
        new_archive = archive
        old_archive_length = len(archive)
        for ind in population:
            if len(new_archive) == 0:
                new_archive.append(ind)
            else:
                should_remove = []
                should_add = True
                for a in new_archive:
                    # this line changed from Inspyred's archiver
                    # if np.all(ind.candidate == a.candidate):
                    if np.all(a.candidate[np.where(ind.candidate == 1)] == 1):
                        should_add = False
                        break
                    elif ind < a:
                        should_add = False
                    # this line changed from Inspyred's archiver
                    # elif ind > a:
                    elif ind > a or np.all(ind.candidate[np.where(a.candidate == 1)] == 1):
                        should_remove.append(a)
                for r in should_remove:
                    new_archive.remove(r)
                if should_add:
                    new_archive.append(ind)

        if len(new_archive) > old_archive_length:

            while True:

                best_overlap = 0
                best_i, best_j = None, None
                best_i_ix, best_j_ix = None, None
                for i in range(len(new_archive) - 1):
                    for j in range(i, len(new_archive)):
                        set_i = set(np.where(new_archive[i].candidate)[0])
                        set_j = set(np.where(new_archive[j].candidate)[0])
                        overlap = len(set_i.intersection(set_j)) / float(max(len(set_i), len(set_j)))
                        if overlap > best_overlap:
                            best_overlap = overlap
                            best_i, best_j = set_i, set_j
                            best_i_ix, best_j_ix = i, j

                if best_overlap > BiclusteringProblem.ARCHIVE_OVERLAP_THRESHOLD:
                    new_candidate = np.zeros(new_archive[0].candidate.shape)
                    new_candidate[list(best_i.union(best_j))] = 1
                    new_individual = Individual(new_candidate, new_archive[0].maximize)
                    new_individual.fitness = args["_ec"].problem.evaluator([new_candidate], args)[0]
                    del new_archive[best_i_ix]
                    del new_archive[best_j_ix]
                    new_archive.append(new_individual)
                else:
                    break

                if len(new_archive) <= 1:
                    break

        return new_archive
