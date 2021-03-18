import sys
import time
import datetime
import numpy as np
from fitting.optimizer import Optimizer
from fitting.ga.generation import Generation
from plots.plot_fitting_output import plot_goodness_of_fit, update_goodness_of_fit_plot, close_goodness_of_fit_plot
    

class GeneticAlgorithm(Optimizer):
    ''' Genetic Algorithm class '''
    
    def __init__(self, name, display_graphics):
        super().__init__(name, display_graphics)
        
    def set_intrinsic_parameters(self, number_of_generations, generation_size, crossover_probability, mutation_probability, parent_selection):
        ''' Set intrinsic parameters of Genetic Algorithm '''
        self.number_of_generations = number_of_generations
        self.generation_size = generation_size
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.parent_selection = parent_selection
    
    def optimize(self, scoring_function, ranges, **kwargs):
        ''' Perform an optimization '''
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
            generation.score_chromosomes(scoring_function, **kwargs)
            # Sort chromosomes according to their score
            generation.sort_chromosomes() 
            # Save the best score in each optimization step
            score_vs_generation.append(generation.chromosomes[0].score)
            # Display graphics
            if self.display_graphics:
                if (i == 0):   
                    fig_score = plot_goodness_of_fit(np.array(score_vs_generation))
                elif ((i > 0) and (i < self.number_of_generations-1)):
                    update_goodness_of_fit_plot(fig_score, np.array(score_vs_generation))
                elif (i == self.number_of_generations-1):
                    close_goodness_of_fit_plot(fig_score)                
            sys.stdout.write('\r')
            sys.stdout.write('Optimization step %d / %d: chi2 = %f' % (i+1, self.number_of_generations, score_vs_generation[i]))
            sys.stdout.flush()
        self.optimized_variables = generation.chromosomes[0].genes
        self.goodness_of_fit = np.array(score_vs_generation)
        time_finish = time.time()
        time_elapsed = str(datetime.timedelta(seconds = time_finish - time_start))
        print('\nThe optimization is finished. Total duration: %s' % (time_elapsed))
        return self.optimized_variables, self.goodness_of_fit