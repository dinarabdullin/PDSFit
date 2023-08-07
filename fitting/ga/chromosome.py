import numpy as np


class Chromosome:
    ''' Chromosome in GA '''

    def __init__(self, bounds):
        self.size = len(bounds)
        self.genes = np.zeros(self.size)
        for i in range(self.size):
            self.genes[i] = self.random_gene(bounds[i][0], bounds[i][1])
        self.score = 0

    def random_gene(self, lower_bound, upper_bound):
        gene = lower_bound + (upper_bound - lower_bound) * np.random.rand(1)
        return gene

    def __lt__(self, chromosome):
        return self.score < chromosome.score