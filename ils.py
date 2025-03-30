import random
import time
import statistics
import utils
from fm_impl import FM
from graph import Graph
import os
import pandas as pd
import numpy as np

class ILS:
    """
    Iterated Local Search using FM 

    Steps:
      1 Generate an initial balanced partition solution 
      2 Run FM local search on it
      3 Keep track of best solutions
      4 Mutate the current best solutions
      5 Run FM again 
      6 Accept if solution is better
      7 Repeat  4 5 6 
    """

    def __init__(self, graph_filename: str, max_iterations=1000, mutation_size=20, random_seed=None):
        self.graph = Graph(graph_filename)
        self.max_iterations = max_iterations
        self.max_cpu_time = 0
        self.spent_cpu_time = 0
        self.n_iterations = 0
        self.mutation_size = mutation_size
        self.random_seed = random_seed
        
        self.best_cut_size = None
        self.best_solution = None
        self.initial_cut_size = None
        self.n_stays_in_local_optimum = 0
        
        self.iteration_data = []  # store data from each iteration
        self.best_cuts = []
        self.mutation_sizes = []
        self.start_time = None
        self.end_time = None
        self.is_adaptive = False
        if self.random_seed is not None:
            random.seed(self.random_seed)

    def _get_mutation_size(self):
        return self.mutation_size
    
    def _update_mutation_operator(self, mutation_size:int, old_cut:int, new_cut:int):
       pass #To be overriden by the adaptive ILS class
    
    def run_ils_max_iter(self, max_iterations:int) -> int:
        """Runs the ILS algorithm for a given number of iterations.
        Args:
            max_iterations (int): Number of iterations to run the ILS algorithm.

        Returns:
            int: best cut size found.
        """
        # run ILS with a maximum number of iterations
        self.max_iterations = max_iterations
        
        self.start_time = time.time()

        #1 
        self.graph.set_random_solution()
        self.initial_cut_size = self.graph.get_cut_size()
        
        # 2 Run FM
        fm_impl = FM(self.graph)
        fm_impl.run_fm()
        self.n_iterations += 1        
        
        # 3 Store best solutions
        self.best_cut_size = self.graph.get_cut_size()
        self.best_solution = self._get_solution()
        
        self.iteration_data.append({
                "iteration": self.n_iterations,
                "best_cut_size_so_far": self.best_cut_size,
                "current_cut_size": self.best_cut_size,
                "same_local_optimum": False
            })
        
        for _ in range(self.max_iterations):
            self._perform_fm()
            pass
        
        self.end_time = time.time()        
        return self.best_cut_size
    
    def run_ils_cpu_time(self, max_cpu_time:int, target_value:int=-1) -> int:
        """Runs the ILS algorithm for a given CPU time.
        The algorithm will stop if the target value is reached or the CPU time exceeds max_cpu_time.
        The target value is -1 by default, which means the algorithm will run until max_cpu_time is reached,
        unless the target value is specified.
        Args:
            max_cpu_time (int): Maximum CPU time in seconds.
            target_value (int, optional): Algorithm stops if a found cut is <= target_value. Defaults to -1.

        Returns:
            int: best cut size found.
        """
        # run ILS with a maximum number of iterations
        self.max_cpu_time = max_cpu_time
        
        self.start_time = time.time()

        #1 
        self.graph.set_random_solution()
        self.initial_cut_size = self.graph.get_cut_size()
        
        # 2 Run FM
        fm_impl = FM(self.graph)
        fm_impl.run_fm()
        self.n_iterations += 1        
        
        # 3 Store best solutions
        self.best_cut_size = self.graph.get_cut_size()
        self.best_solution = self._get_solution()
        
        self.iteration_data.append({
                "iteration": self.n_iterations,
                "best_cut_size_so_far": self.best_cut_size,
                "current_cut_size": self.best_cut_size,
                "same_local_optimum": False
            })
        
        self.spent_cpu_time = 0
        while self.spent_cpu_time < self.max_cpu_time:               
            if self.best_cut_size <= target_value:
                break
            start_time = time.time()     
            self._perform_fm()
            self.spent_cpu_time += time.time() - start_time
            pass
        
        self.end_time = time.time()        
        return self.best_cut_size
    
    def _perform_fm(self)->int:
        """Runs a full FM local search. Handles the mutation and acceptance of new solutions.
        If the solution is better, it will be kept and mutated next time.
        If the solution is worse, it will be reverted to the previous best solution.

        Returns:
            int: new found cut size.
        """
        # Save  best 
        old_best_cut = self.best_cut_size
        old_best_sol = dict(self.best_solution)
        
        # 4 mutate
        self._apply_solution(old_best_sol)
        m_size = self._get_mutation_size()
        self._mutate_solution(m_size)

        # 5 Run FM again from the mutated 
        fm_impl = FM(self.graph)
        fm_impl.run_fm()
        new_cut = self.graph.get_cut_size()
        
        # Call mutation operator updater, if adaptive
        if self.is_adaptive:                
            self._update_mutation_operator(m_size, old_best_cut, new_cut)
        
        #Update the counter.    
        self.n_iterations += 1
        
        if new_cut == old_best_cut:
            self.n_stays_in_local_optimum += 1
        elif new_cut < self.best_cut_size:
            # 6 Accept if it is better
            if self.is_adaptive:
                self.mutation_sizes.append(m_size)
                self.best_cuts.append(new_cut)
            self.best_cut_size = new_cut
            self.best_solution = self._get_solution()
        else:
            # revert to old best
            self._apply_solution(old_best_sol)

        #Track 
        self.iteration_data.append({
            "iteration": self.n_iterations,
            "best_cut_size_so_far": self.best_cut_size,
            "current_cut_size": new_cut,
            "same_local_optimum": new_cut == old_best_cut
        })
        
        return new_cut
    
    # This is replaced by run_ils_cpu_time and run_ils_max_iter methods above.
    # def run_ils(self):
    #     # run ILS 
        
    #     self.start_time = time.time()

    #     #1 
    #     self.graph.set_random_solution()
    #     self.initial_cut_size = self.graph.get_cut_size()
        
    #     # 2 Run FM
    #     fm_impl = FM(self.graph)
    #     fm_impl.run_fm()
    #     self.n_iterations += 1        
        
    #     # 3 Store best solutions
    #     self.best_cut_size = self.graph.get_cut_size()
    #     self.best_solution = self._get_solution()
        
    #     self.iteration_data.append({
    #             "iteration": self.n_iterations,
    #             "best_cut_size_so_far": self.best_cut_size,
    #             "current_cut_size": self.best_cut_size,
    #             "same_local_optimum": False
    #         })

    #     # 4 5 6 
    #     for iteration in range(self.max_iterations):
    #         # Save  best 
    #         old_best_cut = self.best_cut_size
    #         old_best_sol = dict(self.best_solution)
            
    #         # 4 mutate
    #         self._apply_solution(old_best_sol)
    #         m_size = self._get_mutation_size()
    #         self._mutate_solution(m_size)

    #         # 5 Run FM again from the mutated 
    #         fm_impl = FM(self.graph)
    #         fm_impl.run_fm()
    #         new_cut = self.graph.get_cut_size()
            
    #         # Call mutation operator updater, if adaptive
    #         if self.is_adaptive:                
    #             self._update_mutation_operator(m_size, old_best_cut, new_cut)
            
    #         #Update the counter.    
    #         self.n_iterations += 1
            
    #         if new_cut == old_best_cut:
    #             self.n_stays_in_local_optimum += 1
    #         elif new_cut < self.best_cut_size:
    #             # 6 Accept if it is better
    #             if self.is_adaptive:
    #                 self.mutation_sizes.append(m_size)
    #                 self.best_cuts.append(new_cut)
    #             self.best_cut_size = new_cut
    #             self.best_solution = self._get_solution()
    #         else:
    #             # revert to old best
    #             self._apply_solution(old_best_sol)

    #         #Track 
    #         self.iteration_data.append({
    #             "iteration": self.n_iterations,
    #             "best_cut_size_so_far": self.best_cut_size,
    #             "current_cut_size": new_cut,
    #             "same_local_optimum": new_cut == old_best_cut
    #         })

    #     self.end_time = time.time()
    #     # 
    #     return self.best_cut_size

    def _get_solution(self):
        #"gets the partitioning from self.graph into a dict {node_id: partition}."
        sol = {}
        for node_id, node_obj in self.graph.nodes.items():
            sol[node_id] = node_obj.partition
        return sol

    def _apply_solution(self, solution):
        # Apply the solution to the graph
        self.graph.set_solution_explicit(solution)    
    
    def _mutate_solution(self, mutation_size:int):
        # Mutate the current solution
        # uses mutation_size to determine how many nodes to SWAP 
        part0_nodes = [n for n in self.graph.nodes.values() if n.partition == 0]
        part1_nodes = [n for n in self.graph.nodes.values() if n.partition == 1]

        # If the graph is not large enough clamp
        #assert len(part0_nodes) >= self.mutation_size 
        swap_count = min(len(part0_nodes), len(part1_nodes), mutation_size)

        if swap_count < 1:
            return

        # pick random nodes from partition 0
        chosen_from_0 = random.sample(part0_nodes, swap_count)
        # pick random nodes from partition 1
        chosen_from_1 = random.sample(part1_nodes, swap_count)

        # do the swaps
        for node_obj in chosen_from_0:
            self.graph.move_node(node_obj.id)
        for node_obj in chosen_from_1:
            self.graph.move_node(node_obj.id)

    def get_run_statistics(self):
        # Return a summary
       
        total_time = (self.end_time - self.start_time) if (self.start_time and self.end_time) else 0.0
        best_cut_values = [item["best_cut_size_so_far"] for item in self.iteration_data]
        all_cut_values = [item["current_cut_size"] for item in self.iteration_data]
        
        mut_size = self._get_mutation_size()                
        if self.is_adaptive:
            mut_size = statistics.mean(self.mutation_sizes)
            
        return {
            "max_iterations": self.max_iterations,
            "max_cpu_time": self.max_cpu_time,
            "is_adaptive": self.is_adaptive,
            "n_iterations": self.n_iterations,
            "mutation_size": mut_size,
            "initial_cut": self.initial_cut_size,
            "best_cut_size": self.best_cut_size,
            "time_elapsed": total_time,
            "avg_best_cut_size": statistics.mean(best_cut_values) if best_cut_values else None,
            "avg_cut_size": statistics.mean(all_cut_values) if best_cut_values else None,
            #"iteration_log": self.iteration_data,
            "n_stays_in_local_optimum": self.n_stays_in_local_optimum
        }

def run_single_ils(mutation_size:int, max_iterations=10000, random_seed=None)->tuple[int, dict]:
    """Runs the ILS with max_iterations FM passes for once.

    Args:
        mutation_size (int): The mutation size.
        max_iterations (int, optional): Number of FM passes. Defaults to 10000.
        random_seed (_type_, optional): Seed for randomized initial solutions. Might be required for reproducability. Defaults to None.

    Returns:
        tuple[int, dict]: best cut size and the run statistics.
    """
    # 1 Instantiate with desired parameters
    ils = ILS(
        graph_filename="Graph500.txt", 
        max_iterations=max_iterations,    # 
        mutation_size=mutation_size,       # 
        random_seed=random_seed
    )
    
    # 2 Run ILS
    best_cut = ils.run_ils_max_iter(max_iterations)
    
    # 3 Collect run statistics
    stats = ils.get_run_statistics() 
    print(f"ILS - Mutation Size: {mutation_size}. Best Cut: {best_cut}.")        
    
    return best_cut, stats 

def analyze_ils_performance(pickle_folder:str, additional_files:list[str]=None)->pd.DataFrame:
    """Load the ILS experiment results from the pickle files in the given folder (plus any additional files),
    and return a dataframe with the average cut size and mutation size.
    The dataframe will be sorted by mutation size.

    Args:
        pickle_folder (str): The folder containing the ILS results (.pkl files only).
        additional_files (list[str], optional): Any additional ILS results. Defaults to None.

    Returns:
        pd.DataFrame: A dataframe with 3 columns: mutation size, average cut size, and average stays in local optimum.
    The dataframe will be sorted by mutation size.
    """
    # Load all ILS experiment results from pickle files
    data = []
    
    #LLM Prompt: Iterate over the given files in the folder and load the results from each file. Use utils.load_ils_results_from_pickle method.
    files = os.listdir(pickle_folder)
    if additional_files:
        files.extend(additional_files)    
    for file in sorted(files):
        if file.endswith('.pkl'):
            file_path = os.path.join(pickle_folder, file)
            results = utils.load_ils_results_from_pickle(file_path)[1]
            mutation_size = results[0]['mutation_size']
            mean_cut = statistics.mean([r['best_cut_size'] for r in results])
            mean_stays = statistics.mean([r['n_stays_in_local_optimum'] for r in results])
            data.append([mutation_size, mean_cut, mean_stays])
            
            #results_list.append(results[1])
    columns = ['Mutation Size', 'Average Cut Size', 'Stays in Local Optimum']
    df = pd.DataFrame(data=data, columns=columns).sort_values('Mutation Size')
    return df

def compare_results(files:list[str])->pd.DataFrame:  
    """
    Compare the pickle-archived results of the experiments from the given files.
    The best cuts are compared using the Mann-Whitney U test and stored in a dataframe with the p-values of the test.
    The dataframe contains also the average cut size of each experiment. 
    Args:
        files (list[str]): List of file paths to the experiment results.
    Returns:
        pd.DataFrame: A dataframe with the p-values of the Mann-Whitney U test between the experiments and average cut sizes of experiments.
    """    
    expriment_results = {}
    for file in sorted(files):
        if file.endswith('.pkl'):
            results = utils.load_ils_results_from_pickle(file)
            #The summary is the first element of the results.
            algorithm = results[0]['Algorithm']
            mutation_size = results[0]['mutation_size']
            experiment_key = f"{algorithm}-[{mutation_size}]"
            if mutation_size == "N/A":
                experiment_key = algorithm
            results = results[1]# get the details
            expriment_results[experiment_key] = results #store the results of experiment.
    
    # Sort the dictionary by keys
    expriment_results = dict(sorted(expriment_results.items()))    

    #LLM prompt: sort the experiment_results by key. after sorting reinsert the item with key 'MLS' to the beginning.
    # If MLS exists, move it to beginning    
    if 'MLS' in expriment_results:
        mls_value = expriment_results.pop('MLS')
        expriment_results = {'MLS': mls_value, **expriment_results}
    
    experiment_names  = list(expriment_results.keys())
    experiment_values = list(expriment_results.values())
    # Initialize matrix of size len(experiment_names) x len(experiment_names)
    data = np.full((len(experiment_names), len(experiment_names)), np.nan)
    means = np.full(len(experiment_names), np.nan)
    for i in range(len(experiment_names)):
        #name1 = experiment_names[i]
        res1 = experiment_values[i]
        cuts1 = [r['best_cut_size'] for r in res1]
        mean1 = statistics.mean(cuts1)
        means[i] = mean1
        data[i][i] = np.float64(1) #p_value is 1.0 for same data.
        
        for j in range(i + 1,len(experiment_names)):
            #name2 = experiment_names[j]
            res2 = experiment_values[j]
            cuts2 = [r['best_cut_size'] for r in res2]
            #Run the statistical significance test
            _, pvalue = utils.perform_mann_whitney_u_test(cuts1, cuts2)
            data[i][j] = pvalue
            #data[j][i] = pvalue
        #break     
    
    columns = experiment_names
    #LLM prompt: insert the means array as first column to data.    
    data = np.column_stack((means, data))
    # Add 'Mean' to column names at the beginning
    columns = ['Avg Cut Size'] + columns
    df = pd.DataFrame(data=data, columns=columns)
    df.index = experiment_names
    #df = df.round(3)
    return df

if __name__ == "__main__":
    #run_ils_parallel(20, max_iterations=20, runs=10)
    # folder = "pckl/ils_find_mutation_size"
    # files = os.listdir(folder)
    # for i in range(len(files)):
    #     files[i] = os.path.join(folder, files[i])    
    # files = sorted(files)
    # files.insert(0, "pckl/2025-03-26_17-46-39_MLS-runs_10-max_iterations_10000-best_cut_27.8-time_439.552.pkl")
    # df = compare_results(files)    
    pass
