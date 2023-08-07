import numpy as np
from copy import deepcopy
from multiprocessing import Pool
try:
    import mpi4py
    from mpi4py.futures import MPIPoolExecutor
except:
    pass
from mpi.mpi_status import get_mpi
from fitting.ga.chromosome import Chromosome


class Generation:
    ''' Generation in GA '''

    def __init__(self, generation_size):
        self.size = generation_size
        self.size_is_even = True
        # Check if the number of chromosomes is even number
        if (self.size % 2 == 0):
            self.num_pairs = int(self.size / 2)
        else:
            self.num_pairs = int((self.size + 1) / 2)
            self.size_is_even = False
        self.chromosomes = []
        
    def first_generation(self, bounds):
        ''' Creates a first geneteraion '''
        for i in range(self.size):
            chromosome = Chromosome(bounds)
            self.chromosomes.append(chromosome)

    def tournament_selection(self):
        ''' Selects parent chromosomes via tournament selection '''
        index_candidate1 = np.random.random_integers(low=0, high=self.size - 1)
        index_candidate2 = np.random.random_integers(low=0, high=self.size - 1)
        if self.chromosomes[index_candidate1].score < self.chromosomes[index_candidate2].score:
            return index_candidate1
        else:
            return index_candidate2

    def crossover_chromosomes(self, chromosome1, chromosome2, crossover_probability, single_point_crossover_to_uniform_crossover_ratio, exchange_probability):
        ''' Crossovers chromosomes '''
        # Single-point crossover
        if np.random.rand() <= single_point_crossover_to_uniform_crossover_ratio / (1 + single_point_crossover_to_uniform_crossover_ratio):
            if np.random.rand() <= crossover_probability and chromosome1.size > 1:
                genes = deepcopy(chromosome1.genes)
                position = np.random.random_integers(low=1, high=chromosome1.size-1)
                for i in range(position):
                    chromosome1.genes[i] = chromosome2.genes[i]
                    chromosome2.genes[i] = genes[i]
        # Uniform crossover
        else:
            if np.random.rand() <= crossover_probability and chromosome1.size > 1:
                for i in range(chromosome1.size):
                    if np.random.rand() <= exchange_probability:
                        gene = deepcopy(chromosome1.genes[i])
                        chromosome1.genes[i] = chromosome2.genes[i]
                        chromosome2.genes[i] = gene
    
    def mutate_chromosome(self, chromosome, mutation_probability, creep_mutation_to_uniform_mutation_ratio, creep_size, bounds):
        ''' Mutates a chromosome '''
        # Creep mutation
        if np.random.rand() <= creep_mutation_to_uniform_mutation_ratio / (1 + creep_mutation_to_uniform_mutation_ratio):
            for i in range(chromosome.size):
                if (np.random.rand() <= mutation_probability):
                    gene = chromosome.genes[i]
                    lower_bound = gene - creep_size * (bounds[i][1] - bounds[i][0])
                    upper_bound = gene + creep_size * (bounds[i][1] - bounds[i][0])
                    if lower_bound < bounds[i][0]:
                        lower_bound = bounds[i][0]
                    if upper_bound > bounds[i][1]:
                        upper_bound = bounds[i][1]
                    chromosome.genes[i] = chromosome.random_gene(lower_bound, upper_bound)
        # Uniform mutation
        else:
            for i in range(chromosome.size):
                if (np.random.rand() <= mutation_probability):
                    chromosome.genes[i] = chromosome.random_gene(bounds[i][0], bounds[i][1])
        
    def produce_offspring(self, parent_selection, 
                          crossover_probability, single_point_crossover_to_uniform_crossover_ratio, exchange_probability, 
                          mutation_probability, creep_mutation_to_uniform_mutation_ratio, creep_size, bounds):
        ''' Produces a new offspring '''
        offspring = []
        for i in range(self.num_pairs):
            # Select parents
            if parent_selection == 'tournament':
                index_parent1 = self.tournament_selection()
                index_parent2 = self.tournament_selection()
            chromosome1 = deepcopy(self.chromosomes[index_parent1])
            chromosome2 = deepcopy(self.chromosomes[index_parent2])
            # Crossover parents
            self.crossover_chromosomes(chromosome1, chromosome2, crossover_probability, single_point_crossover_to_uniform_crossover_ratio, exchange_probability)
            # Mutate parents
            self.mutate_chromosome(chromosome1, mutation_probability, creep_mutation_to_uniform_mutation_ratio, creep_size, bounds)
            self.mutate_chromosome(chromosome2, mutation_probability, creep_mutation_to_uniform_mutation_ratio, creep_size, bounds)
            # Save new chromosomes
            offspring.append(chromosome1)
            offspring.append(chromosome2)
        if self.size_is_even:
            self.chromosomes = offspring
        else:
            self.chromosomes = offspring[:-1]
            
    def score_chromosomes(self, objective_function):
        ''' Scores chromosomes '''
        variables = []
        for i in range(self.size): 
            variables.append(self.chromosomes[i].genes)
        # for i in range(self.size):
            # self.chromosomes[i].score = objective_function(self.chromosomes[i].genes)        
        run_with_mpi = get_mpi()
        if run_with_mpi:
            with MPIPoolExecutor() as executor:
                result = executor.map(objective_function, variables)
            score = list(result)
        else:
            pool = Pool()
            score = pool.map(objective_function, variables)
            pool.close()
            pool.join() 
        for i in range(self.size):
            self.chromosomes[i].score = score[i]
        
    def sort_chromosomes(self):
        ''' Sorts chromosomes based on their score '''
        self.chromosomes.sort()