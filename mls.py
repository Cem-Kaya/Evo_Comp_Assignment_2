import time
import fm_impl as fm
from graph import Graph
import statistics
import utils
from multiprocessing import Pool

# Create a graph from a file
# Create experiment_results directory if it doesn't exist
#os.makedirs('./experiment_results', exist_ok=True)

class MLS:
    
    def __init__(self, graph_file="Graph500.txt", max_iterations=1000,random_seed=None):
        self.graph_file = graph_file
        self.max_iterations = max_iterations
        self.random_seed = random_seed   
        self.best_cut_size = 1000000
        self.cpu_time_sec = 0
        self.cut_sizes = []
        self.best_cut_sizes = []
        self.results = []
        self.n_iterations = 0
        pass

    def __load_graph(self)->Graph:
        return Graph(self.graph_file)

    def run_single(self)->tuple[dict, list[dict]]:
        graph = self.__load_graph()    
        
        for _ in range(self.max_iterations):
            #Initialize a new random solution.
            graph.set_random_solution(self.random_seed)    
            #Run FM       
            fm_impl = fm.FM(graph)
            cut_size = fm_impl.run_fm()
            self.n_iterations += 1
            #Store the results
            self.cut_sizes.append(cut_size)
            stats = fm_impl.get_run_statistics()        
            self.results.append(stats)            
            if cut_size < self.best_cut_size:
                self.best_cut_size = cut_size
                self.best_cut_sizes.append(cut_size)

        return self.best_cut_size
    
    def run_single_for_cpu_time(self, cpu_time_sec:int)->tuple[dict, list[dict]]:
        self.cpu_time_sec = cpu_time_sec
        graph = self.__load_graph()    
        
        start_time = time.time()
        while (time.time() - start_time) < self.cpu_time_sec:        
            #Initialize a new random solution.
            graph.set_random_solution(self.random_seed)    
            #Run FM       
            fm_impl = fm.FM(graph)
            cut_size = fm_impl.run_fm()
            self.n_iterations += 1
            #Store the results
            self.cut_sizes.append(cut_size)
            stats = fm_impl.get_run_statistics()        
            self.results.append(stats)            
            if cut_size < self.best_cut_size:
                self.best_cut_size = cut_size
                self.best_cut_sizes.append(cut_size)

        return self.best_cut_size

    def get_run_statistics(self)->dict:
        # Return a summary
        results = self.results
        total_time = sum(r['total_elapsed'] for r in results)
        avg_per_fm = statistics.mean(r['total_elapsed'] for r in results)
        initial_cut_values = [item["initial_cut"] for item in results]
        
        return {
            "max_iterations": self.max_iterations,
            "n_iterations": len(results),  
            "best_cut_size": self.best_cut_size,
            "time_elapsed": total_time,
            "avg_cut_size": statistics.mean(self.cut_sizes),
            "avg_best_cut_size": statistics.mean(self.best_cut_sizes),                
            "n_stays_in_local_optimum": "N/A",    
            "n_iterations": self.n_iterations,
            "cpu_time_sec": self.cpu_time_sec,
            "initial_cut_size_avg": statistics.mean(initial_cut_values),
            "initial_cut_size_best": min(initial_cut_values),
            "avg_time_per_fm": avg_per_fm
        }

max_iter=10000
cpu_time=0
graph_filename="Graph500.txt"

def single_run(i, fixed_cpu_time=False):
    
        mls = MLS(
            graph_file=graph_filename, 
            max_iterations=max_iter,            
            random_seed=utils.generate_random_seed()
        )
        mls.cpu_time_sec = cpu_time
        
        start = time.time()
        if fixed_cpu_time:
            best_cut = mls.run_single_for_cpu_time(cpu_time)
        else:
            best_cut = mls.run_single()
        elapsed = round(time.time() - start, 3)
        
        stats = mls.get_run_statistics()        
        print(f"MLS - {i}. Best Cut: {best_cut}. Elapsed: {elapsed}.")        
        return best_cut, stats

def _process_results(results, best,max_iterations, runs, algorithm="MLS", cpu_time_sec=-1):
    avg_best_cut_size = statistics.mean([r['best_cut_size'] for r in results])    
    avg_time_elapsed = statistics.mean([r['time_elapsed'] for r in results])
    
    summary = {"Algorithm":"MLS", 
                "runs": runs,
                "mutation_size": "N/A",
                "max_iterations": max_iterations,      
                "best_cut": best,         
                "avg_best_cut_size": avg_best_cut_size, 
                "avg_time_elapsed": avg_time_elapsed}
    results = list(results)
    results.append(summary)
    
    run_limit = f"max_iterations_{max_iterations}"
    if cpu_time_sec > 0:
        run_limit = f"cpu_time_{cpu_time_sec}"
    
    experiment_name = f"{algorithm}-runs_{runs}-{run_limit}-best_cut_{round(avg_best_cut_size, 2)}-time_{round(avg_time_elapsed, 3)}"
    utils.save_as_pickle(results, experiment_name)
    return best, avg_best_cut_size, results
       
def run_mls(max_iterations=10000, runs:int=10, graph_file="Graph500.txt", cpu_time_sec:int=-1)->tuple[int, float, list[dict]]:   
    global max_iter
    global graph_filename
    global cpu_time
    cpu_time = cpu_time_sec
    graph_filename = graph_file
    max_iter = max_iterations 
    fixed_cpu_time = cpu_time_sec > 0
    
    algorithm = "MLS"
    if fixed_cpu_time:
        algorithm = "MLS-FIXED_CPU"
        
    results = []    
    best = 1000000
    
    for i in range(runs):
        # 2 Run ILS
        best_cut, stats = single_run(i, fixed_cpu_time)
        if best_cut < best:     
            best = best_cut
        # 3 Collect run statistics        
        results.append(stats)     
    
    return _process_results(results, best, max_iterations, runs, algorithm, cpu_time_sec)

# LLM Prompt: Introduce a new run_mls_parallel function that runs the for-loop in run_mls function in parallel.
def run_mls_parallel(max_iterations=10000, runs:int=10, graph_file="Graph500.txt"):   
        global max_iter
        global graph_filename        
        graph_filename = graph_file
        max_iter = max_iterations
        
        with Pool() as pool:
            results_list = pool.map(single_run, range(runs))
        
        best_cuts, results = zip(*results_list)
        best = min(best_cuts)
        
        return _process_results(results, best, max_iterations, runs,"MLS-parallel")

# if __name__ == "__main__":
#     best, avg_best_cut_size, results = run_mls_parallel(max_iterations=20, runs=10)
#     pass