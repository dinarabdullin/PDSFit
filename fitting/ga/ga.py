import sys
import time
import datetime
import numpy as np
from fitting.optimizer import Optimizer
from fitting.ga.generation import Generation
from fitting.ga.plot_score import plot_score, update_score_plot, close_score_plot
    

class GeneticAlgorithm(Optimizer):
    ''' Genetic Algorithm class '''
    
    def __init__(self, name, display_graphics, goodness_of_fit):
        super().__init__(name, display_graphics, goodness_of_fit)
        self.parameter_names = {
            'number_of_runs': 'int', 
            'number_of_generations': 'int', 
            'generation_size': 'int', 
            'crossover_probability': 'float', 
            'mutation_probability': 'float',
            'crossover_probability_increment': 'float', 
            'mutation_probability_increment': 'float', 
            'parent_selection': 'str',
            }
        
    def set_intrinsic_parameters(self, parameters):
        ''' Sets intrinsic parameters of Genetic Algorithm '''
        self.number_of_runs = parameters['number_of_runs'] 
        self.number_of_generations = parameters['number_of_generations'] 
        self.generation_size = parameters['generation_size'] 
        self.crossover_probability = parameters['crossover_probability'] 
        self.mutation_probability = parameters['mutation_probability'] 
        self.crossover_probability_increment = parameters['crossover_probability_increment'] 
        self.mutation_probability_increment = parameters['mutation_probability_increment'] 
        self.parent_selection = parameters['parent_selection'] 
    
    def optimize(self, ranges):
        ''' Performs an optimization '''
        print('\nStarting the optimization via genetic algirithm...')
        time_start = time.time()
        for r in range(self.number_of_runs):
            score_vs_generation = []
            for i in range(self.number_of_generations):     
                if (i == 0):
                    # Create the first generation
                    generation = Generation(self.generation_size)
                    generation.first_generation(ranges)
                else:
                    # Create the next generation
                    crossover_probability = self.crossover_probability + r * self.crossover_probability_increment
                    mutation_probability = self.mutation_probability + r * self.mutation_probability_increment
                    generation.produce_offspring(ranges, self.parent_selection, crossover_probability, mutation_probability)
                # Score the generation
                generation.score_chromosomes(self.objective_function)
                # Sort chromosomes according to their score
                generation.sort_chromosomes() 
                # Save the best score in each optimization step
                score_vs_generation.append(generation.chromosomes[0].score)
                # Display graphics
                if self.display_graphics:
                    if (i == 0):   
                        fig_score = plot_score(np.array(score_vs_generation), self.goodness_of_fit)
                    elif ((i > 0) and (i < self.number_of_generations-1)):
                        update_score_plot(fig_score, np.array(score_vs_generation), self.goodness_of_fit)
                    elif (i == self.number_of_generations-1):
                        close_score_plot(fig_score)                
                sys.stdout.write('\r')
                sys.stdout.write('Run %d / %d, optimization step %d / %d: %s = %f' % (r+1, self.number_of_runs, i+1, self.number_of_generations, self.goodness_of_fit_name, score_vs_generation[i]))
                sys.stdout.flush()
            num_best_solution = 1
            if r == 0:
                self.optimized_variables = generation.chromosomes[0].genes
                self.score = np.array(score_vs_generation)
            else:
                if score_vs_generation[-1] < self.score[-1]:
                    self.optimized_variables = generation.chromosomes[0].genes
                    self.score = np.array(score_vs_generation)
                    num_best_solution = r + 1
        print('\nThe best solution was found in run no. %d (crossover probability %f, mutation_probability %f)' % 
            (num_best_solution, 
             self.crossover_probability + (num_best_solution-1) * self.crossover_probability_increment, 
             self.mutation_probability + (num_best_solution-1) * self.mutation_probability_increment))
        time_finish = time.time()
        time_elapsed = str(datetime.timedelta(seconds = time_finish - time_start))
        print('The optimization is finished. Total duration: %s' % (time_elapsed))
        return self.optimized_variables, self.score