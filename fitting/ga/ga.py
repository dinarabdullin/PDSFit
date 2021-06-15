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
        
    def set_intrinsic_parameters(self, number_of_generations, generation_size, crossover_probability, mutation_probability, parent_selection):
        ''' Sets intrinsic parameters of Genetic Algorithm '''
        self.number_of_generations = number_of_generations
        self.generation_size = generation_size
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.parent_selection = parent_selection
    
    def optimize(self, ranges):
        ''' Performs an optimization '''
        print('\nStarting the optimization via genetic algirithm...')
        time_start = time.time()
        score_vs_generation = []
        for i in range(self.number_of_generations):     
            if (i == 0):
                # Create the first generation
                generation = Generation(self.generation_size)
                generation.first_generation(ranges)
            else:
                # Create the next generation
                generation.produce_offspring(ranges, self.parent_selection, self.crossover_probability, self.mutation_probability)
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
            sys.stdout.write('Optimization step %d / %d: %s = %f' % (i+1, self.number_of_generations, self.goodness_of_fit_name, score_vs_generation[i]))
            sys.stdout.flush()
        self.optimized_variables = generation.chromosomes[0].genes
        self.score = np.array(score_vs_generation)
        time_finish = time.time()
        time_elapsed = str(datetime.timedelta(seconds = time_finish - time_start))
        print('\nThe optimization is finished. Total duration: %s' % (time_elapsed))
        return self.optimized_variables, self.score