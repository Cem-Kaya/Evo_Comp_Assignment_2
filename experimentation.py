import time
from ils import ILS
from ils_adaptive import AdaptiveILS
import ils_adaptive as ila
import ils
import utils
import statistics
import mls

def _run_ils_cpu_time(ils:ILS, max_cpu_time=10*60, target_cut=-1, exp_name="ILS-FIXED_TIME"):
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

def experiment_adaptive_cpu_time(operators:list[int],max_cpu_time=10*60,p_min=0.001, alpha=0.1, beta=0.5, target_cut=2,enable_staging=False, stage_threshold=0.6, stage_cut_limit=40, exp_name="ILS-ADA-FIXED_TIME"):
    print(f"Running Adaptive ILS for {max_cpu_time} seconds...")
    ils = AdaptiveILS("Graph500.txt", operators, p_min, alpha, beta, max_cpu_time)
    ils.use_stage_weight = enable_staging
    ils.stage_threshold = stage_threshold
    ils.stage_cut_limit = stage_cut_limit
    ils.track_history = True
    best_cut, stats = _run_ils_cpu_time(ils, max_cpu_time, target_cut,exp_name)
    print(f"Adaptive ILS - CPU Time: {max_cpu_time}. Best Cut: {best_cut}.")
    return best_cut, stats

def run_ils_ada_staging_param_search():
    operators = [10,20,30,35,40,45,50,55,60,65,70,75,80,85]
    thresholds = [0.3, 0.5, 0.6, 0.7]
    stage_cuts= [30,40,50,60]
    p_min = 0.0001
    alpha = 0.4
    beta = 0.2
    max_iterations = 1000
    runs = 10
    
    combinations = []
    for threshold in thresholds:
        for stage_cut in stage_cuts:
            combinations.append((threshold, stage_cut))    
    
    l = len(combinations)
    for i in range(l):
        threshold, stage_cut = combinations[i]
        print(f"Running Adaptive ILS with staging, threshold: {threshold}, stage limit: {stage_cut}...")
        results = []
        
        for _ in range(runs):
            ils = AdaptiveILS("Graph500.txt", operators, p_min, alpha, beta, max_iterations)
            ils.track_history = True
            ils.use_stage_weight = True
            ils.stage_threshold = threshold
            ils.stage_cut_limit = stage_cut
            
            best_cut = ils.run_ils_max_iter(max_iterations)
            stats = ils.get_run_statistics()
            stats["stage_cut"] = stage_cut
            stats["threshold"] = threshold
            res = [best_cut, stats, ils.mutation_sizes, ils.best_cuts]
            results.append(res)
            print(f"Adaptive ILS - Staging: {threshold}. Limit: {stage_cut}. Best Cut: {best_cut}.")
        # Save results
        experiment_name = f"ILS-ADA-STAGING-threshold_{threshold}-stage_cut_{stage_cut}-runs_{runs}-best_cut_{round(best_cut, 2)}"
        utils.save_as_pickle(results, experiment_name, "pckl/ils-ada-staging-search")
        
       
    pass

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

def run_ils_ada_cpu_time(operators, p_min, alpha, beta, target_cut, max_cpu_time, runs, enable_staging=False, stage_threshold=0.6, stage_cut_limit=40, exp_name="ILS-ADA-CPU_TIME"):
    best_cuts = []
    results = []
    for i in range(runs):
        best_cut, stats = experiment_adaptive_cpu_time(operators, max_cpu_time, p_min, alpha, beta,target_cut, enable_staging, stage_threshold, stage_cut_limit, exp_name)
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
    
    experiment_name = f"{exp_name}-runs_{runs}-max_cpu{max_cpu_time}-best_cut_{round(avg_best_cut_size, 2)}-time_{round(avg_time_elapsed, 3)}"
    utils.save_as_pickle(results, experiment_name)
    
    return best, avg_best_cut_size,results

def run_ils_cpu_time(mutation_size, target_cut, max_cpu_time, runs):
    best_cuts = []
    results = []
    for i in range(runs):
        ils_impl = ils.ILS("Graph500.txt", max_iterations=0, mutation_size=mutation_size)
        best_cut = ils_impl.run_ils_cpu_time(max_cpu_time, target_cut)
        best_cuts.append(best_cut)        
        report = ils_impl.get_run_statistics()        
        results.append(report)
        print(f"ILS - CPU Time [{i}]: {max_cpu_time}. Best Cut: {best_cut}.")       
    
    best = min(best_cuts)
    avg_best_cut_size = statistics.mean(result['best_cut_size'] for result in results)
    avg_time_elapsed = statistics.mean(result['time_elapsed'] for result in results)
    
    summary = {"Algorithm":"ILS-CPU_TIME",
               "mutation_size": mutation_size, 
               "runs": runs,               
               "max_iterations": "N/A",
               "target_cut": target_cut,
               "max_cpu_time": max_cpu_time,               
               "best_cut": best,        
               "avg_best_cut_size": avg_best_cut_size, 
               "avg_time_elapsed": avg_time_elapsed}
    results.append(summary)
    
    experiment_name = f"ILS-CPU_TIME-runs_{runs}-max_cpu{max_cpu_time}-best_cut_{round(avg_best_cut_size, 2)}-time_{round(avg_time_elapsed, 3)}"
    utils.save_as_pickle(results, experiment_name)
    
    return best, avg_best_cut_size,results

def experiment_adaptive_ils(operators, p_min, alpha, beta, max_iterations, runs=1, enable_staging=False, stage_threshold=0.6, stage_cut_limit=40, exp_name="ILS-ADA"):
    results = []    
    best = 10000000
    start = time.time()
    for i in range(runs):
        best_cut, stats = ila.run_ils_adaptive(operators, p_min, alpha, beta, max_iterations, enable_staging=enable_staging, stage_threshold=stage_threshold, stage_cut_limit=stage_cut_limit)
        if best_cut < best:     
            best = best_cut                
        results.append(stats)        
        print(f"{exp_name}-{i} - Best Cut: {best_cut}.")    
    end = time.time()
    
    elapsed_time = end - start
    experiment_name = f"{exp_name}-p{p_min}_a{alpha}_b{beta}_s{enable_staging}-st{stage_threshold}-st{stage_cut_limit}-runs_{runs}-max_iterations_{max_iterations}-best_cut_{best_cut}-time_{round(elapsed_time, 3)}"
    utils.save_as_pickle(results, experiment_name)
    return best, results

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
    #run_ils_ada_25()
    pass