from ils import ILS
from ils_adaptive import AdaptiveILS
import ils_adaptive as ila
import ils
import utils
import statistics
import mls

def run_ils_cpu_time(ils:ILS, max_cpu_time=10*60, target_cut=-1, exp_name="ILS-FIXED_TIME"):
    #1 Run ILS
    best_cut = ils.run_ils_cpu_time(max_cpu_time, target_cut)
    
    # 3 Collect run statistics
    stats = ils.get_run_statistics() 
    print(f"ILS - CPU Time: {max_cpu_time}. Best Cut: {best_cut}.")        
    
    results = [stats]
    if ils.is_adaptive:
        results.append(ils.mutation_sizes)
        results.append(ils.best_cuts)
        results.append({"p_min": ils.p_min, "alpha": ils.alpha, "beta": ils.beta,"use_stage_weights": ils.use_stage_weight})
    
    experiment_name = f"{exp_name}-max_cpu_{max_cpu_time}-stage_weights:{ils.use_stage_weight}-n_iterations_{ils.n_iterations}-best_cut_{round(best_cut, 2)}"
    utils.save_as_pickle(results, experiment_name)
    return best_cut, stats

def experiment_adaptive_cpu_time(operators:list[int],max_cpu_time=10*60,p_min=0.001, alpha=0.1, beta=0.5, target_cut=2, exp_name="ILS-FIXED_TIME"):
    print(f"Running Adaptive ILS for {max_cpu_time} seconds...")
    ils = AdaptiveILS("Graph500.txt", operators, p_min, alpha, beta, max_cpu_time)
    ils.track_history = True
    best_cut, stats = run_ils_cpu_time(ils, max_cpu_time, target_cut,exp_name)
    print(f"Adaptive ILS - CPU Time: {max_cpu_time}. Best Cut: {best_cut}.")
    return best_cut, stats

def run_adaptive_ils_param_search():
    operators = [30,35,40,45,50,55,60,65,70,75,80,85]
    #2000 iterations 86.26 seconds
    max_iterations = 2000
    p_mins = [0.04,0.01,0.001,0.0001]
    alphas = [0.1,0.2,0.4,0.6]
    betas = [0.1,0.2,0.3,0.5]

    print("Running Adaptive ILS with parameter search - parallel...")
    results = ila.run_parameter_search(operators=operators, p_mins=p_mins, alphas=alphas, betas=betas, max_iterations=max_iterations)
    print("Finished running Adaptive ILS with parameter search.")
    
def run_ils(mutation_size:int, max_iterations=10000, runs:int=10, exp_name="ILS"):
    
    results = []    
    best = 10000000
    
    for i in range(runs):
        best_cut, stats = ils.run_single_ils(mutation_size, max_iterations)       
        if best_cut < best:     
            best = best_cut                
        results.append(stats)        
        print(f"ILS -{i}- Mutation Size: {mutation_size}. Best Cut: {best_cut}.")        
    
    #get the average best_cut_size from the results
    avg_best_cut_size = statistics.mean([r['best_cut_size'] for r in results])    
    #get the average time elapsed from the results
    avg_time_elapsed = statistics.mean([r['time_elapsed'] for r in results])
    #Insert both metrics into results
    summary = {"Algorithm":"ILS", 
               "runs": runs,
               "mutation_size": mutation_size,
               "max_iterations": max_iterations,      
               "best_cut": best,        
               "avg_best_cut_size": avg_best_cut_size, 
               "avg_time_elapsed": avg_time_elapsed}
    results.append(summary)
    
    experiment_name = f"{exp_name}-mutation_{mutation_size}-runs_{runs}-max_iterations_{max_iterations}-best_cut_{round(avg_best_cut_size, 2)}-time_{round(avg_time_elapsed, 3)}"
    utils.save_as_pickle(results, experiment_name)
    return best, avg_best_cut_size,results

def run_ils_parallel(mutation_size:int, max_iterations=10000, runs:int=10):
    # Parallel run of ILS
    from concurrent.futures import ProcessPoolExecutor
    
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(ils.run_single_ils, mutation_size, max_iterations) for _ in range(runs)]
        results_list = [future.result() for future in futures]
    
    best_cuts, results = zip(*results_list)
    results = list(results)
    
    best = min(best_cuts)
    avg_best_cut_size = statistics.mean(result['best_cut_size'] for result in results)
    avg_time_elapsed = statistics.mean(result['time_elapsed'] for result in results)
    
    summary = {"Algorithm":"ILS", 
               "runs": runs,
               "mutation_size": mutation_size,
               "max_iterations": max_iterations,      
               "best_cut": best,        
               "avg_best_cut_size": avg_best_cut_size, 
               "avg_time_elapsed": avg_time_elapsed}
    results.append(summary)
    
    experiment_name = f"ILS-mutation_{mutation_size}-runs_{runs}-max_iterations_{max_iterations}-best_cut_{round(avg_best_cut_size, 2)}-time_{round(avg_time_elapsed, 3)}"
    utils.save_as_pickle(results, experiment_name)
    
    return best, avg_best_cut_size,results

def run_ils_ada_25():
    operators = [10,20,30,40,45,50,55,60,65,70]
    p_min = 0.0001
    alpha = 0.4
    beta = 0.2
    target_cut = 2
    max_cpu_time=450
    
    # Parallel run of ILS
    #from concurrent.futures import ProcessPoolExecutor
    
    #best_cut, stats = experiment_adaptive_cpu_time(operators, max_cpu_time, p_min, alpha, beta,target_cut)
    runs = 25
    best_cuts = []
    results = []
    for i in range(runs):
        best_cut, stats = experiment_adaptive_cpu_time(operators, max_cpu_time, p_min, alpha, beta,target_cut, exp_name="ILS-ADA")
        best_cuts.append(best_cut)
        results.append(stats)
        print(f"Adaptive ILS - CPU Time: {max_cpu_time}. Best Cut: {best_cut}.")
    
    # with ProcessPoolExecutor() as executor:
    #     futures = []
    #     max_workers = 10
    #     for i in range(0, runs, max_workers):
    #         batch = [executor.submit(experiment_adaptive_cpu_time, operators, max_cpu_time, p_min, alpha, beta,target_cut) 
    #             for _ in range(i, min(i + max_workers, runs))]
    #         futures.extend(batch)
    #     results_list = [future.result() for future in futures]
    
    # best_cuts, results = zip(*results_list)
    # results = list(results)
    
    best = min(best_cuts)
    avg_best_cut_size = statistics.mean(result['best_cut_size'] for result in results)
    avg_time_elapsed = statistics.mean(result['time_elapsed'] for result in results)
    
    summary = {"Algorithm":"ILS-ADA", 
               "runs": runs,
               "mutation_size": "N/A",
               "max_iterations": "N/A",
               "p_min": p_min,
               "alpha": alpha,
               "beta": beta,
               "target_cut": target_cut,
               "max_cpu_time": max_cpu_time,
               "operators": operators,      
               "best_cut": best,        
               "avg_best_cut_size": avg_best_cut_size, 
               "avg_time_elapsed": avg_time_elapsed}
    results.append(summary)
    
    experiment_name = f"ILS-ADA-runs_{runs}-max_cpu{max_cpu_time}-best_cut_{round(avg_best_cut_size, 2)}-time_{round(avg_time_elapsed, 3)}"
    utils.save_as_pickle(results, experiment_name)
    
    return best, avg_best_cut_size,results

if __name__ == "__main__":
    # operators = [30,35,40,45,50,55,60,65,70,75,80,85]
    # #2000 iterations 86.26 seconds
    # max_iterations = 20
    # p_mins = [0.04,0.01,0.001,0.0001]
    # alphas = [0.1,0.2,0.4,0.6]
    # betas = [0.1,0.2,0.3,0.5]
    # print("Running Adaptive ILS with parameter search - parallel...")
    # results = run_parameter_search(operators=operators, p_mins=p_mins, alphas=alphas, betas=betas, max_iterations=max_iterations)
    # print("Finished running Adaptive ILS with parameter search.")
    #run_adaptive_ils_param_search()
    
    #operators = [30,35,40,45,50,55,60,65,70,75,80,85]    
    # operators = [10,20,30,40,45,50,55,60,65,70]
    # p_min = 0.0001
    # alpha = 0.4
    # beta = 0.2
    # target_cut = 2
    # max_cpu_time=10*60
    # best_cut, stats = experiment_adaptive_cpu_time(operators, max_cpu_time, p_min, alpha, beta,target_cut)
    run_ils_ada_25()
    pass