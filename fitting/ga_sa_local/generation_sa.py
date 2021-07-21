import numpy as np
from copy import deepcopy
from multiprocessing import Pool
from fitting.ga_sa_local.chromosome_sa import Chromosome


class Generation:
    ''' Generation class with Simulated Annealing '''

    def __init__(self, generation_size):
        self.size = generation_size
        self.chromosomes = []
        self.sa_chromosomes = []
        self.T0 = 0
        self.T_index = 0
        self.count = 0
        
    def first_generation(self, bounds):
        ''' Creates first geneteraion '''
        for i in range(self.size):
            chromosome = Chromosome(bounds)
            self.chromosomes.append(chromosome)
    
    def produce_sa_chromosomes(self, bounds):
        ''' Creates random chromosomes for Simulated Annealing '''
        self.sa_chromosomes = []
        for i in range(self.size):
            chromosome = Chromosome(bounds)
            self.sa_chromosomes.append(chromosome)    

    def tournament_selection(self):
        ''' Selects parent chromosomes via tournament selection '''
        index_candidate1 = np.random.random_integers(low=0, high=self.size-1)
        index_candidate2 = np.random.random_integers(low=0, high=self.size-1)
        if self.chromosomes[index_candidate1].score < self.chromosomes[index_candidate2].score:
            return index_candidate1
        else:
            return index_candidate2
            
    def random_selection(self):
        ''' Selects parent chromosomes randomly '''
        return np.random.random_integers(low=0, high=self.size-1)
            
    def crossover_chromosomes(self, chromosome1, chromosome2, crossover_probability):
        ''' Crossovers chromosomes '''
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
        ''' Mutates a chromosome '''
        for i in range(chromosome.size):
            if (np.random.rand() <= mutation_probability):
                chromosome.genes[i] = chromosome.random_gene(bounds[i][0], bounds[i][1])
        
    def produce_offspring(self, bounds, parent_selection, elitism, crossover_probability, mutation_probability):
        ''' Produces a new offspring '''
        # Choose the potential parents
        potential_parents = []
        for i in range(self.size):
            if parent_selection == 'tournament':
                index_potential_parent = self.random_selection()
                potential_parent = self.chromosomes[index_potential_parent]
            potential_parents.append(potential_parent)
        # Choose the parents via Simulated Annealing
        parents = []
        self.count = 0
        for i in range(self.size):
            score = potential_parents[i].score
            sa_score = self.sa_chromosomes[i].score
            if sa_score <= score:
                parents.append(self.sa_chromosomes[i])
            else:
                sa_probability = np.power(self.T_index, (score - sa_score)/self.T0)
                if np.random.rand() <= sa_probability:
                    parents.append(self.sa_chromosomes[i])
                    self.count +=1
                else:
                    parents.append(potential_parents[i])
        # Produce an offspring
        # Elitism
        offspring = []
        if elitism:
            num_elite = 1
            elite_chromosome = deepcopy(self.chromosomes[0])
            offspring.append(elite_chromosome)
        else:
            num_elite = 0
        # Check if the number of nonelite parents is even
        even = True
        num_pairs = 0
        if ((self.size-num_elite) % 2 == 0):
            num_pairs = int((self.size-num_elite) / 2)
        else:
            num_pairs = int((self.size-num_elite+1) / 2)
            even = False
        # Select / crossover / mutate nonelite parents
        for i in range(num_pairs):
            # Select parents
            index_parent1 = self.tournament_selection()
            index_parent2 = self.tournament_selection()
            chromosome1 = deepcopy(parents[index_parent1])
            chromosome2 = deepcopy(parents[index_parent2])
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
            
    def score_chromosomes(self, objective_function):
        ''' Scores chromosomes '''
        variables = []
        for i in range(self.size): 
            variables.append(self.chromosomes[i].genes)
        for i in range(self.size):
            variables.append(self.sa_chromosomes[i].genes)
        self.pool = Pool()
        score = self.pool.map(objective_function, variables)
        self.pool.close()
        self.pool.join()
        for i in range(self.size):
            self.chromosomes[i].score = score[i]
            self.sa_chromosomes[i].score = score[self.size+i]
    
    def sort_chromosomes(self):
        ''' Sorts chromosomes based on their score '''
        self.chromosomes.sort()