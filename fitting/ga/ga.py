import sys
import time
import datetime
import numpy as np
from scipy import optimize
from fitting.optimizer import Optimizer
from fitting.ga.generation import Generation
    

class GeneticAlgorithm(Optimizer):
    """Genetic algorithm (GA) followed by Nelder-Mead algorithm (NMA)."""
    
    def __init__(self, name):
        super().__init__(name)
        self.intrinsic_parameter_names = {
            "number_of_runs": "int", 
            "generation_size": "int", 
            "parent_selection": "str",
            "crossover_probability": "float", 
            "mutation_probability": "float", 
            "single_point_crossover_to_uniform_crossover_ratio": "float",
            "creep_mutation_to_uniform_mutation_ratio": "float",
            "exchange_probability": "float",
            "creep_size": "float",
            "maximum_number_of_generations": "int", 
            "maximum_number_of_generations_with_constant_score": "int",
            "accepted_score_variation": "float",
            "maximal_number_of_nma_iterations": "int"
            }
        self.run_number = 1
        self.ga_iteration = 1
        self.nma_iteration = 1
        self.xv, self.yv, self.y_vs_iter = [], [], []
    
    
    def set_intrinsic_parameters(self, intrinsic_parameters):
        """Set intrinsic parameters of GA."""
        self.num_runs = intrinsic_parameters["number_of_runs"] 
        self.generation_size = intrinsic_parameters["generation_size"] 
        self.parent_selection = intrinsic_parameters["parent_selection"]
        self.crossover_prob = intrinsic_parameters["crossover_probability"] 
        self.mutation_prob = intrinsic_parameters["mutation_probability"] 
        self.single_point_vs_uniform_crossover = intrinsic_parameters["single_point_crossover_to_uniform_crossover_ratio"] 
        self.creep_vs_uniform_mutation = intrinsic_parameters["creep_mutation_to_uniform_mutation_ratio"] 
        self.exchange_prob = intrinsic_parameters["exchange_probability"] 
        self.creep_size = intrinsic_parameters["creep_size"]
        self.max_num_generations = intrinsic_parameters["maximum_number_of_generations"]
        self.max_num_generations_const_score = intrinsic_parameters["maximum_number_of_generations_with_constant_score"]
        self.accepted_score_variation = intrinsic_parameters["accepted_score_variation"]
        self.max_num_nma_iter = intrinsic_parameters["maximal_number_of_nma_iterations"] 
    
    
    def optimize(self, bounds):
        """ Performs an optimization """
        sys.stdout.write(
            "\n########################################################################\
            \n#  Fitting via genetic algorithm (GA) and Nelder-Mead algorithm (NMA)  #\
            \n########################################################################\n")
        sys.stdout.write("\n")
        sys.stdout.flush()
        time_start = time.time()
        optimized_models, score_all_runs = [], []
        for r in range(self.num_runs):
            self.run_number = r + 1
            scores = []
            # Genetic algorithm
            for i in range(self.max_num_generations):     
                self.ga_iteration = i + 1
                if i == 0:
                    # Create a first generation
                    generation = Generation(self.generation_size)
                    generation.first_generation(bounds)
                else:
                    # Create next generation
                    generation.produce_offspring(
                        self.parent_selection,
                        self.crossover_prob,
                        self.single_point_vs_uniform_crossover,
                        self.exchange_prob,
                        self.mutation_prob,
                        self.creep_vs_uniform_mutation,
                        self.creep_size,
                        bounds
                        )
                # Score the generation
                generation.score_chromosomes(self.scoring_function)
                # Sort chromosomes according to their score
                generation.sort_chromosomes() 
                # Save the best score for each iteration
                scores.append(generation.chromosomes[0].score)
                # Print status
                sys.stdout.write("\r")
                sys.stdout.write(
                    "Run {0:d} / {1:d}, GA optimization step {2:d} / {3:d}: {4:s} = {5:<20.1f}".format(
                        self.run_number, 
                        self.num_runs, 
                        self.ga_iteration, 
                        self.max_num_generations,
                        self.goodness_of_fit,
                        scores[i]
                        )
                    )
                sys.stdout.flush()
                # If the score did not change by "accepted_score_variation" in the last
                # "maximum_number_of_generations_with_constant_score" generations, finish the optimization
                if i >= self.max_num_generations_const_score:
                    mean_score = np.mean(scores[i - self.max_num_generations_const_score:i])
                    score_change = np.absolute(scores[i] - mean_score) / scores[i]
                    if score_change < self.accepted_score_variation:
                        break
            optimized_model = generation.chromosomes[0].genes
            
            # Nelder-Mead algorithm
            self.nma_iteration = 1
            self.xv, self.yv, self.y_vs_iter = [], [], []
            bounds_tuple = [tuple(v) for v in bounds]
            result = optimize.minimize(
                self.modified_scoring_function, 
                x0 = optimized_model,
                method = "Nelder-Mead", 
                # bounds = bounds_tuple,
                args = (bounds_tuple),
                callback = self.callback, 
                options = {
                    "maxiter": self.max_num_nma_iter, 
                    "maxfev": self.max_num_nma_iter,
                    "adaptive": True
                    }
                )
            optimized_model = np.array(result.x)
            scores = scores + self.y_vs_iter
            
            # Best model
            optimized_models.append(optimized_model)
            score_all_runs.append(np.array(scores))
            if r == 0:
                index_best_model = 0
            else:
                if scores[-1] < score_all_runs[index_best_model][-1]:
                    index_best_model = r          
        sys.stdout.write(
            "\nThe best model was found in optimization run no. {0:d}\n".format(index_best_model + 1)
            )
        time_elapsed = str(datetime.timedelta(seconds = time.time() - time_start))
        sys.stdout.write("\nThe optimization is finished. Duration: %s\n" % (time_elapsed))
        sys.stdout.flush()
        return optimized_models, index_best_model, score_all_runs
        
    
    def modified_scoring_function(self, x, bounds):
        """Modified scoring function that stores the scoring function values.""" 
        y = self.scoring_function(x)
        self.xv.append(x)
        if not self.solution_is_within_bounds(x, bounds):
            y = 2 * y
        self.yv.append(y)
        return y
    
    
    def solution_is_within_bounds(self, solution, bounds):
        """Check whether the solution is within the specified bounds."""
        status = True
        for i in range(len(solution)):
            if (solution[i] < bounds[i][0]) or (solution[i] > bounds[i][1]):
                status = False
        return status
    
    
    def callback(self, xk, *_):
        """ Callback function.
        The third argument "*_" makes sure that it still works when the
        optimizer calls the callback function with more than one argument."""
        xk = np.atleast_1d(xk)
        for i, x in reversed(list(enumerate(self.xv))):
            x = np.atleast_1d(x)
            if np.allclose(x, xk, rtol = 0, atol = 1e-8):
                break
        self.y_vs_iter.append(self.yv[i])
        sys.stdout.write("\r")
        sys.stdout.write(
            "Run {0:d} / {1:d}, NMA optimization step {2:d} / {3:d}: {4:s} = {5:<20.1f}".format(
                self.run_number, 
                self.num_runs, 
                self.nma_iteration, 
                self.max_num_nma_iter, 
                self.goodness_of_fit, 
                self.y_vs_iter[-1]
                )
            )
        sys.stdout.flush()
        self.nma_iteration += 1