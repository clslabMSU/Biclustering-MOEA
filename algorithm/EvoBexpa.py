from bisect import insort_left
from copy import deepcopy
import numpy as np
from random import Random
from sys import stdout
from time import time
from typing import Dict, List, NewType

from algorithm.NSGA_II import BiclusteringProblem

from Bicluster import Bicluster
from BiclusterSet import BiclusterSet
from Dataset import Dataset
from Metrics import GeneVariance_fast as GeneVariance, VEt_fast as VEt, VEt as VEt_old, bicluster_volume


# extend tuple class for only comparing first entry in operator<
class Individual(tuple):

    def __init__(self, tup):
        super().__init__()

    def __new__(cls, tup):
        return super(Individual, cls).__new__(cls, tuple(tup))

    def __lt__(self, other: "Individual") -> bool:
        return self[0] < other[0]


Population = NewType("Population", Dict[int, Individual])


class EvoBexpa:

    dataset = None
    dataset_VEt = None

    w_s, w_ov, w_var = None, None, None
    w_gene, w_sample = None, None
    pop_size = None

    weights, nb = None, None

    MAX_GENS = 1500
    MAX_GENS_NO_PROGRESS = 150

    P_MUTATE = 0.01

    P_CHOOSE_MUTATION_UNIFORM = 0.1

    def __init__(self):
        pass

    @staticmethod
    def cluster(dataset: Dataset,
                num_bics: int = 5,
                w_s: float = 5,
                w_ov: float = 5,
                w_var: float = 0.1,
                w_gene: float = 0.25,
                w_sample: float = 0.5,
                pop_size: int = 100
                ) -> BiclusterSet:

        EvoBexpa.dataset = dataset
        EvoBexpa.dataset_VEt = VEt(dataset.matrix)
        EvoBexpa.w_s = w_s
        EvoBexpa.w_ov = w_ov
        EvoBexpa.w_var = w_var
        EvoBexpa.w_gene = w_gene
        EvoBexpa.w_sample = w_sample
        EvoBexpa.pop_size = pop_size

        biclusters = BiclusterSet()
        EvoBexpa.weights = np.zeros(dataset.matrix.shape)
        for nb in range(1, num_bics + 1):

            EvoBexpa.nb = nb

            # use Bexpa to generate bicluster
            bic = EvoBexpa.Bexpa(dataset)

            # append new bicluster to bicluster list
            biclusters.append(bic)

            # update weights
            EvoBexpa.weights[np.ix_(bic.genes(), bic.samples())] += 1
        return biclusters


    @staticmethod
    def Bexpa(dataset: Dataset) -> Bicluster:
        problem = BiclusteringProblem(dataset, [EvoBexpa.fitness])

        population = [] # type: List[Individual]

        for i in range(EvoBexpa.pop_size):
            chromosome = problem.generator(Random(), {})
            bic = Bicluster.from_chromosome(chromosome, EvoBexpa.dataset)
            bic.chromosome = chromosome
            insort_left(population, (Individual((EvoBexpa.fitness(bic), bic))))

        #population.join()

        generation = 0
        last_progress_gen = 0
        best_ind_ix = None
        best_fit = None

        start_time = time()

        while generation < EvoBexpa.MAX_GENS and generation - last_progress_gen < EvoBexpa.MAX_GENS_NO_PROGRESS:

            generation += 1

            mutated_best_ind = EvoBexpa.mutate(population[0])
            next_population = [population[0]] # type: List[Individual]
            insort_left(next_population, mutated_best_ind)
            next_population_size = 2

            stdout.write("\r\t\tRep: %d\t\tGeneration: %d\t\tBest Fitness: %.4f\t\tLast Progress: %d\t\tElapsed time: %d min %d sec" % \
                         (EvoBexpa.nb, generation, population[0][0], last_progress_gen, (time() - start_time)/60, (time() - start_time) % 60))
            stdout.flush()

            # generate 80% of individuals via crossover and mutation
            while next_population_size < 0.8*EvoBexpa.pop_size:
                child1, child2 = EvoBexpa.crossover(EvoBexpa.tournament_selection(population, 3), EvoBexpa.tournament_selection(population, 3))
                if np.random.rand() < EvoBexpa.P_MUTATE:
                    child1 = EvoBexpa.mutate(child1)
                if np.random.rand() < EvoBexpa.P_MUTATE:
                    child2 = EvoBexpa.mutate(child2)
                insort_left(next_population, child1)
                insort_left(next_population, child2)
                next_population_size += 2

            # generate 20% of individuals via mutation only
            while next_population_size < EvoBexpa.pop_size:
                individual = EvoBexpa.tournament_selection(population, 3)
                if np.random.rand() < EvoBexpa.P_MUTATE:
                    individual = EvoBexpa.mutate(individual)
                insort_left(next_population, individual)
                next_population_size += 1

            population = deepcopy(next_population)

            # TODO - max or min? I think min
            best_fitness, best = population[0]

            if best_fit is None or best_fitness < best_fit:
                best_fit = best_fitness
                best_ind_ix = deepcopy(best)
                last_progress_gen = generation

        print()
        values = EvoBexpa.chromosome_to_ndarray(best_ind_ix.chromosome)
        genes = np.where(best_ind_ix.chromosome[:dataset.matrix.shape[0]])[0]
        samples = np.where(best_ind_ix.chromosome[dataset.matrix.shape[0]:])[0]

        return Bicluster(genes, samples, values)


    @staticmethod
    def overlap(bicluster: Bicluster, weights: np.ndarray, nb: int) -> float:
        denom = len(bicluster.genes()) * len(bicluster.samples()) * (nb - 1)
        if denom == 0:
            return 1.
        return sum(weights[np.ix_(bicluster.genes(), bicluster.samples())].flatten()) / denom


    #@staticmethod
    #def gene_variance(bicluster: Bicluster) -> float:
    #    row_means = np.mean(bicluster._values_, axis=1)
    #    I, J = bicluster._values_.shape
    #    # TODO - vectorize this
    #    return 1./(I*J) * sum([sum([(bicluster._values_[i, j] - row_means[i])**2 for j in range(J)]) for i in range(I)])


    @staticmethod
    def fitness(bicluster: Bicluster) -> float:
        return bicluster.VEt()/EvoBexpa.dataset_VEt + \
               EvoBexpa.w_s * bicluster_volume(bicluster.values(), [EvoBexpa.w_gene, EvoBexpa.w_sample]) + \
               EvoBexpa.w_ov * EvoBexpa.overlap(bicluster, EvoBexpa.weights, EvoBexpa.nb) + \
               EvoBexpa.w_var / (1 + GeneVariance(bicluster.values()))


    @staticmethod
    def chromosome_to_ndarray(chromosome: np.ndarray) -> np.ndarray:
        genes = np.where(chromosome[:EvoBexpa.dataset.matrix.shape[0]] == 1)[0]
        samples = np.where(chromosome[EvoBexpa.dataset.matrix.shape[0]:] == 1)[0]
        return EvoBexpa.dataset.matrix[np.ix_(genes, samples)]


    @staticmethod
    def simple_mutation(chromosome: np.ndarray) -> np.ndarray:
        # idk what they mean by "simple mutation", so I just did flip 1 bit with probability
        if np.random.rand() < EvoBexpa.P_MUTATE:
            i = np.random.randint(0, chromosome.shape[0])
            chromosome[i] = 1 - chromosome[i]
        return chromosome


    @staticmethod
    def uniform_mutation(chromosome: np.ndarray) -> np.ndarray:
        # they didn't say whether two mutation operators use different probabilities, so they use the same here
        for i in range(chromosome.shape[0]):
            if np.random.rand() < EvoBexpa.P_MUTATE:
                chromosome[i] = 1 - chromosome[i]
        return chromosome


    @staticmethod
    def mutate(bicluster: Individual) -> Individual:
        if np.random.rand() < EvoBexpa.P_CHOOSE_MUTATION_UNIFORM:
            result = EvoBexpa.uniform_mutation(bicluster[1].chromosome)
        else:
            result = EvoBexpa.simple_mutation(bicluster[1].chromosome)
        new_bic = Bicluster.from_chromosome(result, EvoBexpa.dataset)
        return Individual((EvoBexpa.fitness(new_bic), new_bic))


    @staticmethod
    def uniform_crossover(parent1: Bicluster, parent2: Bicluster) -> (Bicluster, Bicluster):
        child1, child2 = parent1.chromosome, parent2.chromosome
        for i in range(parent1.chromosome.shape[0]):
            if np.random.rand() < 0.5:
                child1[i] = parent1.chromosome[i]
                child2[i] = parent2.chromosome[i]
            else:
                child1[i] = parent2.chromosome[i]
                child2[i] = parent1.chromosome[i]
        return Bicluster.from_chromosome(child1, EvoBexpa.dataset), Bicluster.from_chromosome(child2, EvoBexpa.dataset)


    @staticmethod
    def n_point_crossover(mom: Bicluster, dad: Bicluster, num_crossover_points: int) -> (Bicluster, Bicluster):
        """
        N-point crossover code borrowed and adapted from Inspyred
        """
        num_cuts = min(len(mom.chromosome) - 1, num_crossover_points)
        cut_points = np.random.choice(list(range(1, len(mom.chromosome))), num_cuts, replace=False)
        cut_points.sort()
        bro = deepcopy(dad.chromosome)
        sis = deepcopy(mom.chromosome)
        normal = True
        for i, (m, d) in enumerate(zip(mom.chromosome, dad.chromosome)):
            if i in cut_points:
                normal = not normal
            if not normal:
                bro[i] = m
                sis[i] = d
                normal = not normal
        return Bicluster.from_chromosome(bro, EvoBexpa.dataset), Bicluster.from_chromosome(sis, EvoBexpa.dataset)


    @staticmethod
    def crossover(parent1: Individual, parent2: Individual) -> (Individual, Individual):
        type_of_crossover = np.random.randint(1, 4)
        if type_of_crossover <= 2:
            child1, child2 = EvoBexpa.n_point_crossover(parent1[1], parent2[1], type_of_crossover)
        else:
            child1, child2 = EvoBexpa.uniform_crossover(parent1[1], parent2[1])
        return Individual((EvoBexpa.fitness(child1), child1)), Individual((EvoBexpa.fitness(child2), child2))


    @staticmethod
    def tournament_selection(population: List[Individual], tournament_size: int) -> Individual:
        contenders = np.random.choice(list(range(EvoBexpa.pop_size)), tournament_size, replace=False)
        # TODO - tournament selection on an ordered list can be reduced to sampling from the pdf of min(X1, X2, X3) where Xi ~ DiscreteUnif(0, pop_size), Xi != Xj, might be faster this way
        return population[min(contenders)]

