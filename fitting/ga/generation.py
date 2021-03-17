import numpy as np
from copy import deepcopy
from multiprocessing import Pool
from functools import partial
from fitting.ga.chromosome import Chromosome
from fitting.scoring_function import scoring_function


class Generation:
    ''' Generation class '''

    def __init__(self, generation_size):
        self.size = generation_size
        self.chromosomes = []
        
    def first_generation(self, bounds):
        ''' Create first geneteraion '''
        for i in range(self.size):
            chromosome = Chromosome(bounds)
            self.chromosomes.append(chromosome)

    def tournament_selection(self):
        ''' Select parent chromosomes via tournament selection '''
        index_candidate1 = np.random.random_integers(low=0, high=self.size - 1)
        index_candidate2 = np.random.random_integers(low=0, high=self.size - 1)
        if self.chromosomes[index_candidate1].score < self.chromosomes[index_candidate2].score:
            return index_candidate1
        else:
            return index_candidate2

    def crossover_chromosomes(self, chromosome1, chromosome2, crossover_probability):
        ''' Crossover chromosomes '''
        if np.random.rand() <= crossover_probability:
            # Store the genes of chromosome1
            genes = deepcopy(chromosome1.genes)
            # Choose a random crossover position
            position = np.random.random_integers(low=1, high=chromosome1.size - 2)
            # Crossover
            for i in range(position):
                chromosome1.genes[i] = chromosome2.genes[i]
                chromosome2.genes[i] = genes[i]

    def mutate_chromosome(self, chromosome, mutation_probability, bounds):
        ''' Crossover a chromosome '''
        for i in range(chromosome.size):
            if (np.random.rand() <= mutation_probability):
                chromosome.genes[i] = chromosome.random_gene(bounds[i][0], bounds[i][1])
        
    def produce_offspring(self, bounds, parent_selection, crossover_probability, mutation_probability):
        ''' Produce a new offspring '''
        # Check if the number of chromosomes is even
        even = True
        num_pairs = 0
        if (self.size % 2 == 0):
            num_pairs = int(self.size / 2)
        else:
            num_pairs = int((self.size + 1) / 2)
            even = False
        # Select the pairs of parents and produce an offspring
        offspring = []
        for i in range(num_pairs):
            # Select parents
            if parent_selection == 'tournament':
                index_parent1 = self.tournament_selection()
                index_parent2 = self.tournament_selection()
            chromosome1 = deepcopy(self.chromosomes[index_parent1])
            chromosome2 = deepcopy(self.chromosomes[index_parent2])
            # Crossover parents
            self.crossover_chromosomes(chromosome1, chromosome2, crossover_probability)
            # Mutate parents
            self.mutate_chromosome(chromosome1, mutation_probability, bounds)
            self.mutate_chromosome(chromosome2, mutation_probability, bounds)
            # Save new chromosomes
            offspring.append(chromosome1)
            offspring.append(chromosome2)
        if even:
            self.chromosomes = offspring
        else:
            self.chromosomes = offspring[:-1]
            
    def score_chromosomes(self, scoring_function, **kwargs):
        func = partial(scoring_function, **kwargs)
        variables = []
        for i in range(self.size): 
            variables.append(self.chromosomes[i].genes)
        self.pool = Pool()
        score = self.pool.map(func, variables)
        self.pool.close()
        self.pool.join()
        for i in range(self.size):
            self.chromosomes[i].score = score[i]
        # for i in range(self.size):
            # self.chromosomes[i].score = scoring_function(self.chromosomes[i].genes, **kwargs)
        
    def sort_chromosomes(self):
        self.chromosomes.sort()