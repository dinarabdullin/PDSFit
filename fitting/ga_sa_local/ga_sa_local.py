import sys
import time
import datetime
import numpy as np
from scipy import optimize
from fitting.optimizer import Optimizer
from fitting.ga_sa_local.generation_sa import Generation
from fitting.ga.plot_score import plot_score, update_score_plot, close_score_plot
    

class GeneticAlgorithmWithSimulatedAnnealingAndLocalSolver(Optimizer):
    ''' Hybrid Genetic Algorithm with Simulated Annealing  '''
    
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
            'elitism': 'int',
            'max_number_of_permanent_generations': 'int',
            'nelder_mead_maxiter': 'int',
            }
        self.score_local = []
        self.x_values = []
        self.y_values = []
        self.count = 0
        
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
        self.elitism = parameters['elitism'] 
        self.max_number_of_permanent_generations = parameters['max_number_of_permanent_generations']
        self.nelder_mead_maxiter = parameters['nelder_mead_maxiter'] 
        
    def optimize(self, ranges):
        ''' Performs an optimization '''
        print('\nStarting the optimization via Genetic Algorithm / Simulated Annealing...')
        time_start = time.time()
        for r in range(self.number_of_runs):
            score_vs_generation = []
            best_chromosome_vs_generation = []
            for i in range(self.number_of_generations):     
                if i == 0:
                    # Create the first generation
                    generation = Generation(self.generation_size)
                    generation.first_generation(ranges)
                    generation.produce_sa_chromosomes(ranges)
                else:
                    # Create the next generation
                    crossover_probability = self.crossover_probability + r * self.crossover_probability_increment
                    mutation_probability = self.mutation_probability + r * self.mutation_probability_increment
                    generation.produce_offspring(ranges, self.parent_selection, self.elitism, crossover_probability, mutation_probability)
                    generation.produce_sa_chromosomes(ranges)
                # Score the generation
                generation.score_chromosomes(self.objective_function)
                # Sort chromosomes according to their score
                generation.sort_chromosomes()
                # Save the best score in each optimization step
                score_vs_generation.append(generation.chromosomes[0].score)
                # Save the best chromosome in each optimization step
                best_chromosome_vs_generation.append(generation.chromosomes[0])
                # Set the temperature for Simulated Annealing
                if i == 0:
                    generation.T0 = generation.chromosomes[-1].score
                    generation.T_index = 1   
                else:
                    generation.T_index += 1
                if i > self.max_number_of_permanent_generations:
                    current_best_chromosome = best_chromosome_vs_generation[-1]
                    past_best_chromosome = best_chromosome_vs_generation[-self.max_number_of_permanent_generations]
                    if current_best_chromosome == past_best_chromosome:
                        generation.T0 = generation.chromosomes[-1].score
                        generation.T_index = 1               
                # Display graphics
                if self.display_graphics:
                    if i == 0:   
                        fig_score = plot_score(np.array(score_vs_generation), self.goodness_of_fit)
                    elif (i > 0) and (i < self.number_of_generations-1):
                        update_score_plot(fig_score, np.array(score_vs_generation), self.goodness_of_fit)
                    elif i == self.number_of_generations-1:
                        close_score_plot(fig_score)                
                sys.stdout.write("\r")
                sys.stdout.write('Run %d / %d, opt. step %d / %d: %s = %f (SA: T0 = %d, i = %d, %d accepted sets)   ' %
                    (r+1, self.number_of_runs, i+1, self.number_of_generations, self.goodness_of_fit_name, score_vs_generation[i], generation.T0, generation.T_index, generation.count))
                sys.stdout.flush()
            if r == 0:
                self.optimized_variables = generation.chromosomes[0].genes
                self.score = np.array(score_vs_generation)
            else:
                if score_vs_generation[-1] < self.score[-1]:
                    self.optimized_variables = generation.chromosomes[0].genes
                    self.score = np.array(score_vs_generation)
        time_finish = time.time()
        time_elapsed = str(datetime.timedelta(seconds = time_finish - time_start))
        print('\nThe optimization is finished. Total duration: %s' % (time_elapsed))
        
        print('\nStarting the optimization via Nelder-Mead algirithm...')
        time_start = time.time()
        bounds = [tuple(x) for x in ranges] 
        result = optimize.minimize(self.modified_objective_function, x0=self.optimized_variables, args=(), method='Nelder-Mead', bounds=bounds, callback=self.callback, 
            options={'maxiter': self.nelder_mead_maxiter, 'maxfev': None, 'initial_simplex': None, 'xatol': 0.0001,'fatol': 0.0001, 'adaptive': True})     
        self.optimized_variables = np.array(result.x)
        self.score = np.append(self.score, self.score_local)
        time_finish = time.time()
        time_elapsed = str(datetime.timedelta(seconds = time_finish - time_start))
        print('\nThe optimization is finished. Total duration: %s' % (time_elapsed))

        return self.optimized_variables, self.score
    
    def modified_objective_function(self, x, *args):
        ''' Modified objective function that stores the function value ''' 
        y = self.objective_function(x, *args)
        self.x_values.append(x)
        self.y_values.append(y)
        return y
    
    def callback(self, xk, *_):
        ''' 
        Callback function that can be used by optimizers of scipy.optimize.
        The third argument "*_" makes sure that it still works when the
        optimizer calls the callback function with more than one argument.
        '''
        xk = np.atleast_1d(xk)
        for i, x in reversed(list(enumerate(self.x_values))):
            x = np.atleast_1d(x)
            if np.allclose(x, xk):
                break           
        self.score_local.append(self.y_values[i])
        sys.stdout.write('\r')
        sys.stdout.write('Optimization step %d / %d: %s = %f' % (self.count+1, self.nelder_mead_maxiter, self.goodness_of_fit_name, self.y_values[i]))
        sys.stdout.flush()
        self.count += 1