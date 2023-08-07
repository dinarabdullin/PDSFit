import sys
import time
import datetime
import numpy as np
from scipy import optimize
from fitting.optimizer import Optimizer
from fitting.ga.generation import Generation
    

class GeneticAlgorithm(Optimizer):
    ''' Genetic algorithm extended by Nelder-Mead algorithm '''
    
    def __init__(self, name):
        super().__init__(name)
        self.parameter_names = {
            'number_of_trials':                                     'int', 
            'generation_size':                                      'int', 
            'parent_selection':                                     'str',
            'crossover_probability':                                'float', 
            'mutation_probability':                                 'float',
            'crossover_probability_increment':                      'float', 
            'mutation_probability_increment':                       'float', 
            'single_point_crossover_to_uniform_crossover_ratio':    'float',
            'creep_mutation_to_uniform_mutation_ratio':             'float',
            'exchange_probability':                                 'float',
            'creep_size':                                           'float',
            'maximum_number_of_generations':                        'int', 
            'maximum_number_of_generations_with_constant_score':    'int',
            'accepted_score_variation':                             'float',
            'maximal_number_of_nma_iterations':                     'int'
            }
        self.x_values = []
        self.y_values = []
        self.score_local = []
        self.run_number = 1
        self.ga_iteration_number = 1
        self.nma_iteration_number = 1        
        
    def set_intrinsic_parameters(self, parameters):
        ''' Sets the intrinsic parameters of GA '''
        self.number_of_trials = parameters['number_of_trials'] 
        self.generation_size = parameters['generation_size'] 
        self.parent_selection = parameters['parent_selection']
        self.crossover_probability = parameters['crossover_probability'] 
        self.mutation_probability = parameters['mutation_probability'] 
        self.crossover_probability_increment = parameters['crossover_probability_increment'] 
        self.mutation_probability_increment = parameters['mutation_probability_increment'] 
        self.single_point_crossover_to_uniform_crossover_ratio = parameters['single_point_crossover_to_uniform_crossover_ratio'] 
        self.creep_mutation_to_uniform_mutation_ratio = parameters['creep_mutation_to_uniform_mutation_ratio'] 
        self.exchange_probability = parameters['exchange_probability'] 
        self.creep_size = parameters['creep_size']
        self.maximum_number_of_generations = parameters['maximum_number_of_generations']
        self.maximum_number_of_generations_with_constant_score = parameters['maximum_number_of_generations_with_constant_score']
        self.accepted_score_variation = parameters['accepted_score_variation']
        self.maximal_number_of_nma_iterations = parameters['maximal_number_of_nma_iterations'] 
    
    def optimize(self, ranges):
        ''' Performs an optimization '''
        sys.stdout.write('\n########################################################################\
                          \n#  Fitting via genetic algorithm (GA) and Nelder-Mead algorithm (NMA)  #\
                          \n########################################################################\n\n')
        sys.stdout.flush()
        time_start = time.time()
        for r in range(self.number_of_trials):
            self.run_number = r + 1 
            optimized_variables_single_run = []
            score_single_run = []
            
            # Genetic algorithm
            score_vs_generation = []
            for i in range(self.maximum_number_of_generations):     
                self.ga_iteration_number = i + 1
                if i == 0:
                    # Create the first generation
                    generation = Generation(self.generation_size)
                    generation.first_generation(ranges)
                else:
                    # Create the next generation
                    crossover_probability = self.crossover_probability + r * self.crossover_probability_increment
                    mutation_probability = self.mutation_probability + r * self.mutation_probability_increment
                    generation.produce_offspring(self.parent_selection, 
                                                 crossover_probability, self.single_point_crossover_to_uniform_crossover_ratio, self.exchange_probability,
                                                 mutation_probability, self.creep_mutation_to_uniform_mutation_ratio, self.creep_size, ranges)
                # Score the generation
                generation.score_chromosomes(self.objective_function)
                # Sort chromosomes according to their score
                generation.sort_chromosomes() 
                # Save the best score in each optimization step
                score_vs_generation.append(generation.chromosomes[0].score)
                # Display graphics                
                sys.stdout.write('\r')
                sys.stdout.write('Run {0:d} / {1:d}, GA optimization step {2:d} / {3:d}: {4:s} = {5:.1f}          '.format(
                    self.run_number, self.number_of_trials, self.ga_iteration_number, self.maximum_number_of_generations, self.goodness_of_fit_name, score_vs_generation[i]))
                sys.stdout.flush()
                # If the score did not change by 'accepted_score_variation' in the last 'maximum_number_of_generations_with_constant_score' 
                # generations, finish the optimization
                if (i >= self.maximum_number_of_generations_with_constant_score):
                    mean_score = np.mean(score_vs_generation[i-self.maximum_number_of_generations_with_constant_score:i])
                    score_change = np.absolute(score_vs_generation[i] - mean_score) / score_vs_generation[i]
                    if (score_change < self.accepted_score_variation):
                        break
            optimized_variables_single_run = generation.chromosomes[0].genes
            score_single_run = np.array(score_vs_generation)
            
            # Nelder-Mead algorithm
            self.nma_iteration_number = 1
            self.score_local = []
            self.x_values = []
            self.y_values = []
            bounds = [tuple(x) for x in ranges]
            result = optimize.minimize(self.modified_objective_function, 
                                       x0=optimized_variables_single_run,
                                       method='Nelder-Mead', 
                                       bounds=bounds,
                                       args=(bounds),
                                       callback=self.callback, 
                                       options={'maxiter': self.maximal_number_of_nma_iterations, 
                                                'maxfev': self.maximal_number_of_nma_iterations,
                                                'adaptive': True})
            optimized_variables_single_run = np.array(result.x)
            score_single_run = np.append(score_single_run, self.score_local)
            
            # Statistics
            self.optimized_variables.append(optimized_variables_single_run)
            self.score.append(score_single_run)
            if r == 0:
                self.idx_best_solution = 0
            else:
                if score_single_run[-1] < self.score[self.idx_best_solution][-1]:
                    self.idx_best_solution = r          
        sys.stdout.write('\nThe best solution was found in optimization run no. {0:d}\n'.format(self.idx_best_solution + 1))
        time_elapsed = str(datetime.timedelta(seconds = time.time() - time_start))
        sys.stdout.write('\nThe optimization is finished. Duration: %s\n' % (time_elapsed))
        sys.stdout.flush()
        return self.optimized_variables, self.score, self.idx_best_solution
    
    def modified_objective_function(self, x, bounds):
        ''' Modified objective function that stores the function value ''' 
        y = self.objective_function(x)
        self.x_values.append(x)
        if not self.solution_is_within_bounds(x, bounds):
            y = 2*y
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
            if np.allclose(x, xk, rtol=0, atol=1e-8):
                break
        self.score_local.append(self.y_values[i])
        sys.stdout.write('\r')
        sys.stdout.write('Run {0:d} / {1:d}, NMA optimization step {2:d} / {3:d}: {4:s} = {5:.1f}          '.format(
            self.run_number, self.number_of_trials, self.nma_iteration_number, self.maximal_number_of_nma_iterations, self.goodness_of_fit_name, self.y_values[i]))
        sys.stdout.flush()
        self.nma_iteration_number += 1
        
    def solution_is_within_bounds(self, solution, bounds):
        ''' Check that the solution of the Nelder-Mead algorithm is within the specified bounds'''
        status = True
        for i in range(len(solution)):
            if (solution[i] < bounds[i][0]) or (solution[i] > bounds[i][1]):
                status = False
        return status