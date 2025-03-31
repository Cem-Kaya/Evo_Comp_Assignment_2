import time
from ils import ILS
import numpy as np
import utils

class AdaptiveILS(ILS):    
    
    def __init__(self, graph_filename: str, mutation_operators:list[int], p_min:float, alpha=0.1, beta=0.1 , max_iterations=1000, random_seed=None):
        """Implementation of the extended version of ILS algorithm with adaptive pursuit.
        The mutation size is dynamically adapted based on the performance of the operators over time.
                
        Args:
            graph_filename (str): Name of the graph file.
            mutation_operators (list[int]): A list of candidate mutation sizes.
            p_min (float): Minimum possible probability for an operator. This is necessary in non-stationary environments, 
                as it is not desired to have a probability of 0 for an operator.
            alpha (float, optional): Learning rate for reward estimates. Defaults to 0.1.
            beta (float, optional): Learning rate for operator probabilities. Defaults to 0.1.
            max_iterations (int, optional): Number of FM iterations. This values is ignored if the algorithm run in fixed cpu time mode. Defaults to 1000.
            random_seed (float, optional): For initializing new random partitionings. May be required for reproducability. Defaults to None.

        Raises:
            ValueError: Raise error if p_min is greater than or equal to 1/K, where K is the number of mutation operators.
        """
        super().__init__(graph_filename, max_iterations, -1, random_seed)
        self.is_adaptive = True
        self.K = len(mutation_operators) 
        
        #check if p_min is correct. It must be less than 1/K
        if p_min >= 1 / self.K:
            raise ValueError("p_min must be less than 1/K.")
        
        #Initialize probability vector with equal probabilities
        probability_vector = np.array([1.0 / self.K] * self.K)        
        #Initialize the reward estimates for each operator. Assign 1.0 to all operators.
        reward_estimates = np.array([1.0] * self.K)            
        # Parameters for adaptive mutation size
        self.mutation_operators = np.zeros((self.K, 3),dtype=object)
        self.min_operator = min(mutation_operators)
        self.max_operator = max(mutation_operators)
        self.operator_indices = np.full(max(mutation_operators) + 1,dtype=int,fill_value=-1)        
        for i in range(len(mutation_operators)):
            self.operator_indices[mutation_operators[i]] = i
            self.mutation_operators[i] = [mutation_operators[i], probability_vector[i], reward_estimates[i]]            
        
        #self.best_operator_history = []        
        self.reward_history = []
        self.operator_history = []
        #self.a_star_history = []
        #Minimum possible probability for an operator. This is necessary in non-stationary environments.
        # It is not possible to have a probability of 0 for an operator.
        self.p_min = p_min         
        #Maximum possible probability for an operator. In case all others are at p_min.
        self.p_max = 1 - (self.K - 1)* p_min   
          
        #Learning rate for the reward estimates.
        self.alpha = alpha
        #Learning rate for the probabilities.
        self.beta = beta
        self.track_history = False
        self.use_stage_weight = False
        self.stage_threshold = 0.6
        self.stage_cut_limit = 50
        
    def _get_mutation_size(self):
        #Return the operator with the highest probability
        idx = np.argmax(self.mutation_operators[:, 1]) # The probability of the best operator is at column 1.
        return self.mutation_operators[idx][0] #return the mutation size of the best operator.
    
    def stage_weight(self,mutation_size):  
        """
        Experimental: try to favor higher mutation sizes in the early stage of the algorithm and lower mutation sizes in the late stage.
        """      
        stage = self._get_stage() #get the current stage of the algorithm
        # Normalize mutation size to [0,1]
        norm_size = (mutation_size - self.min_operator) / (self.max_operator - self.min_operator)
        
        # Early: favor large (1 - norm_size), Late: favor small (norm_size)
        weight = (1 - stage) * norm_size + stage * (1 - norm_size)      
        return weight
    
    def _get_stage(self):
        if(self.max_cpu_time > 0):
            stage = self.spent_cpu_time/self.max_cpu_time
        else:
            stage = self.n_iterations / self.max_iterations
        return stage

    def _update_mutation_operator(self, mutation_size:int, old_cut:int, new_cut:int):
        r = old_cut - new_cut #actual reward
        op_index = self.operator_indices[mutation_size] #index of the operator in the mutation_operators array
        a = self.mutation_operators[op_index] #current operator
        q = a[2] #current reward estimate: Qa(t)        
        #Update the reward estimate of current operator, Qas(t+1) = Qas(t) + alpha * (Ras(t) - Qas(t)) 
        #a[2] = q + self.alpha * (r - q) 
        # if self.use_stage_weight:
        #     stage_weight = self.stage_weight(a[0])
        #     a[2] = q + self.alpha * stage_weight * (r - q)
        # else:
        a[2] = q + self.alpha * (r - q)
        
        #select the best operator. It is the one with the highest reward estimate.
        best_index = np.argmax(self.mutation_operators[:, 2])
        a_star = self.mutation_operators[best_index]
        # Update the probability of the best operator at Pa*(t+1):
        # Pa∗ (t + 1) = Pa∗ (t) + β[Pmax − Pa∗ (t)]
        # this will increase the probability of the best operator (pursue)
        a_star[1] = a_star[1] + self.beta * (self.p_max - a_star[1])
        if self.track_history:
            self.reward_history.append(r)
            #self.operator_history.append(np.array(a))
            #self.a_star_history.append(np.array(a_star))
            #Append the operator with highest probability to the history
            #self.best_operator_history.append(np.array(self.mutation_operators[np.argmax(self.mutation_operators[:, 1])])) 
        
        #update the probabilities of the other operators
        stage = self._get_stage()
        for i in range(self.K):
            if i == best_index:
                continue # skip the best operator
            
            a = self.mutation_operators[i]            
            # Update the probability of the current operator:
            # Pa(t + 1) = Pa(t) + β[Pmin − Pa(t)]
            # this will penalize the probability of other operators, to pursue the best operator.
            if self.use_stage_weight:
                if stage < self.stage_threshold and a[0] < self.stage_cut_limit:
                    a[1] = self.p_min
            else:
                a[1] = a[1] + self.beta * (self.p_min - a[1])
        
        if self.track_history:
            self.operator_history.append(np.array(self.mutation_operators))
        pass

def _run_adaptive_single(operators, p_min, alpha, beta, max_iterations):    
    start = time.time()
    ils = AdaptiveILS("Graph500.txt", operators, p_min, alpha, beta, max_iterations)
    ils.track_history = True
    best_cut = ils.run_ils_max_iter(max_iterations=max_iterations)
    end = time.time()        
    
    return best_cut, ils, end - start

def run_ils_adaptive(operators, p_min, alpha, beta, max_iterations, enable_staging=False, stage_threshold=0.6, stage_cut_limit=40) -> tuple[int, dict]:
    """Runs adaptive ILS with the given parameters.

    Args:
        operators (_type_): Mutation sizes
        p_min (_type_): Minimum probability for an operator
        alpha (_type_): Reward estimate learning rate
        beta (_type_): Probability learning rate
        max_iterations (_type_): FM iterations
        enable_staging (bool, optional): Enable staging, it suppresses low mutation sizes in the first phase of the process. Defaults to False.
        stage_threshold (float, optional): The progress percentage until low mutation sizes are suppressed. Defaults to 0.6. 
        stage_cut_limit (int, optional): The mutation size limit to suppress for staging. Defaults to 40.

    Returns:
        tuple[int, dict]: best cut and execution report.
    """
    ils = AdaptiveILS("Graph500.txt", operators, p_min, alpha, beta, max_iterations)
    ils.track_history = True
    if enable_staging:
        ils.use_stage_weight = True
        ils.stage_threshold = stage_threshold
        ils.stage_cut_limit = stage_cut_limit
        
    best_cut = ils.run_ils_max_iter(max_iterations=max_iterations)
    stats = ils.get_run_statistics()
    stats["p_min"] = p_min
    stats["alpha"] = alpha
    stats["beta"] = beta
    stats["enable_staging"] = enable_staging
    stats["stage_cut"] = stage_cut_limit
    stats["threshold"] = stage_threshold
    stats["best_cut"] = best_cut
    stats["mutation_sizes"] = ils.mutation_sizes
    stats["best_cuts"] = ils.best_cuts
    return best_cut, stats

def run_parameter_search(operators:list[int], p_mins:list[float], alphas:list[float], betas:list[float], max_iterations:int=2000):
    """Performs a parameter search for the adaptive ILS algorithm.
    It runs the algorithm with different combinations of parameters and saves the results.
    The results are saved in a pickle file.
    Number of combinations = len(p_mins) * len(alphas) * len(betas)
    Args:
        operators (list[int]): Investigated mutation sizes
        p_mins (list[float]): Investigated minimum probabilities
        alphas (list[float]): Investigated learning rates for reward estimates
        betas (list[float]): Investigated learning rates for probabilities
        max_iterations (int, optional): Number of FM runs. Defaults to 2000.
    """
    combinations = []
    for p_min in p_mins:
        for alpha in alphas:
            for beta in betas:
                combinations.append((p_min,alpha,beta))
    
    # Parallel run of ILS
    from concurrent.futures import ProcessPoolExecutor
    
    #LLM Prompt: run the ILS with the parameters in parallel for combinations.
    with ProcessPoolExecutor() as executor:
        futures = []
        for p_min, alpha, beta in combinations:
            futures.append(executor.submit(_run_adaptive_single, operators, p_min, alpha, beta, max_iterations))
        results_list = [future.result() for future in futures]
    
    results = []
    for (p_min, alpha, beta), (best_cut, ils, elapsed_time) in zip(combinations, results_list):
        best_history = [ils.mutation_sizes , ils.best_cuts]
        res = {"p_min": p_min, "alpha": alpha, "beta": beta, "best_cut": best_cut, "elapsed_time": elapsed_time, "best_operator": ils.mutation_sizes[-1] ,"sizes_cuts":best_history ,"reward_history": ils.reward_history,  "operator_history": ils.operator_history}
        results.append(res)
        
    # for p_min, alpha, beta in combinations:
    #     best_cut, ils, elapsed_time = _run_adaptive_single(operators, p_min, alpha, beta, max_iterations)
    #     results[(p_min,alpha,beta)] = (best_cut, elapsed_time, ils._get_mutation_size(), ils.best_operator_history,ils.reward_history)
        #print(f"Best cut: {best_cut} Time: {elapsed_time:.2f} seconds")
    #save the results
    #Result format: (p_min, alpha, beta) : (best_cut, elapsed_time, mutation_size, reward_history, operator_history, best_operator_history, a_star_history)
    utils.save_as_pickle(results, f"adaptive_ils_parameter_search_iterations-{max_iterations}", folder="./pckl")

def _load_results_from_folder(folder:str):
    """Load the results from the folder.
    Args:
        folder (str): Folder where the results are saved.
    """
    import os
    import pickle
    results = []
    for filename in os.listdir(folder):
        if filename.endswith(".pkl"):
            with open(os.path.join(folder, filename), "rb") as f:
                results.append(pickle.load(f))
    return results
if __name__ == "__main__":
    operators = [10,20,30,35,40,45,50,55,60,65,70,75,80,85]
    # #2000 iterations 86.26 seconds
    max_iterations = 1000
    # p_mins = [0.04,0.01,0.001,0.0001]
    # alphas = [0.1,0.2,0.4,0.6]
    # betas = [0.1,0.2,0.3,0.5]
    # print("Running Adaptive ILS with parameter search - parallel...")
    # results = run_parameter_search(operators=operators, p_mins=p_mins, alphas=alphas, betas=betas, max_iterations=max_iterations)
    # print("Finished running Adaptive ILS with parameter search.")
    p_min = 0.0001
    alpha = 0.4
    beta = 0.2
    
    start = time.time()
    ils = AdaptiveILS("Graph500.txt", operators, p_min, alpha, beta, max_iterations)
    ils.track_history = True
    ils.use_stage_weight = True
    ils.stage_threshold = 0.6
    ils.stage_cut_limit = 50
    best_cut = ils.run_ils_max_iter(max_iterations=max_iterations)
    end = time.time()
    pass