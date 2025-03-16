import fm_impl as fm
from graph import Graph
import json
import datetime
import os

# Create a graph from a file
# Create experiment_results directory if it doesn't exist
os.makedirs('./experiment_results', exist_ok=True)
    
def load_graph():
    return Graph("Graph500.txt")

def run_mls(runs=10000)->list[dict]:
    graph = load_graph()    
    results = []
    
    for _ in range(runs):
        #Initialize a new random solution.
        graph.set_random_solution()
        current_cut = graph.get_cut_size()
        fm_impl = fm.FM(graph)
        fm_impl.run_fm()
        stats = fm_impl.get_run_statistics()
        stats['initial_cut'] = current_cut
        results.append(stats)

    # LLM prompt: serialize the results as json to a file. File name format is yyyy-mm-dd_HH-MM-SS_MLS.txt
    # Generate filename with current timestamp
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = f'./experiment_results/{timestamp}_MLS.txt'

    # Write results to file
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)
    return results

def summarize_results(results:list[dict]):
    # This is the content of the each dictionary in the results list:
    # "fm_runs": number of runs until convergence,
    # "run_times": a list of the run times of each run,
    # "total_elapsed": total time elapsed,
    # "average_elapsed": average time elapsed,
    # "cut_size": best cut size found,
    # "partition_1": a list of node ids in partition 1,
    # "partition_2": a list of node ids in partition 2,
    # "initial_cut_size": initial cut size
    
    # LLM Prompt: Summarize the results. Return a dictionary with the following keys:
    # "Average Runs": average number of runs until convergence,
    # "Average Elapsed (full FM)": average time elapsed (average of total_elapsed),
    # "Average Elapsed (single run)": average time elapsed (average of run_times),
    # "Average Cut Size": average cut_size found,
    # "Best Cut Size": best cut size (minimum cut_size),
    # "Worst Cut Size": worst cut size (maximum cut_size),
    # "Average Initial Cut Size": average initial_cut_size
    # "Best Initial Cut Size": best initial cut (minimum initial_cut_size)
    # "Worst Initial Cut Size": worst initial cut (maximum initial_cut_size)
    summary = {
        "Total Runs": len(results),
        "Average Runs": sum(r['fm_runs'] for r in results) / len(results),
        "Average Elapsed (full FM)": sum(r['total_elapsed'] for r in results) / len(results),
        "Average Elapsed (single run)": sum(sum(r['run_times']) / len(r['run_times']) for r in results) / len(results),
        "Total Elapsed": sum(r['total_elapsed'] for r in results),
        "Average Cut Size": sum(r['cut_size'] for r in results) / len(results),
        "Best Cut Size": min(r['cut_size'] for r in results),
        "Worst Cut Size": max(r['cut_size'] for r in results),
        "Average Initial Cut Size": sum(r['initial_cut'] for r in results) / len(results),
        "Best Initial Cut Size": min(r['initial_cut'] for r in results),
        "Worst Initial Cut Size": max(r['initial_cut'] for r in results)
    }
    return summary
    
    pass

def deserialize_results(filename)->list[dict]:
    #LLM Prompt: Deserialize the results from the file and return the list of dictionaries
    with open (filename, 'r') as f:
        data = json.load(f)
        return [dict(item) for item in data]
    