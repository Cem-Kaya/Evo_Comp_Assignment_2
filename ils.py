import random
import time
import statistics
import utils
from fm_impl import FM
from graph import Graph

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
        self.mutation_size = mutation_size
        self.random_seed = random_seed
        
        self.best_cut_size = None
        self.best_solution = None
        self.initial_cut_size = None
        self.n_stays_in_local_optimum = 0
        
        self.iteration_data = []  # store data from each iteration
        self.start_time = None
        self.end_time = None
        
        if self.random_seed is not None:
            random.seed(self.random_seed)

    def run_ils(self):
        # run ILS 
        
        self.start_time = time.time()

        #1 
        self.graph.set_random_solution()
        self.initial_cut_size = self.graph.get_cut_size()
        
        # 2 Run FM
        fm_impl = FM(self.graph)
        fm_impl.run_fm()
        
        # 3 Store best solutions
        self.best_cut_size = self.graph.get_cut_size()
        self.best_solution = self._get_solution()

        # 4 5 6 
        for iteration in range(self.max_iterations):
            # Save  best 
            old_best_cut = self.best_cut_size
            old_best_sol = dict(self.best_solution)

            # 4 mutate
            self._apply_solution(old_best_sol)  
            self._mutate_solution()

            # 5 Run FM again from the mutated 
            fm_impl = FM(self.graph)
            fm_impl.run_fm()
            new_cut = self.graph.get_cut_size()

            if new_cut == old_best_cut:
                self.n_stays_in_local_optimum += 1
            elif new_cut < self.best_cut_size:
                # 6 Accept if it is better
                self.best_cut_size = new_cut
                self.best_solution = self._get_solution()
            else:
                # revert to old best
                self._apply_solution(old_best_sol)

            #Track 
            self.iteration_data.append({
                "iteration": iteration + 1,
                "best_cut_size_so_far": self.best_cut_size,
                "current_cut_size": new_cut,
                "same_local_optimum": new_cut == old_best_cut
            })

        self.end_time = time.time()
        # 
        return self.best_cut_size

    def get_run_statistics(self):
        # Return a summary
       
        total_time = (self.end_time - self.start_time) if (self.start_time and self.end_time) else 0.0
        cut_values = [item["best_cut_size_so_far"] for item in self.iteration_data]
        return {
            "max_iterations": self.max_iterations,
            "mutation_size": self.mutation_size,
            "initial_cut": self.initial_cut_size,
            "best_cut_size": self.best_cut_size,
            "time_elapsed": total_time,
            "average_cut_in_iterations": statistics.mean(cut_values) if cut_values else None,
            "iteration_log": self.iteration_data,
            "n_stays_in_local_optimum": self.n_stays_in_local_optimum
        }

    def _get_solution(self):
        #"gets the partitioning from self.graph into a dict {node_id: partition}."
        sol = {}
        for node_id, node_obj in self.graph.nodes.items():
            sol[node_id] = node_obj.partition
        return sol

    def _apply_solution(self, solution):
        # Apply the solution to the graph
        self.graph.set_solution_explicit(solution)

    def _mutate_solution(self):
        # Mutate the current solution
        # uses mutation_size to determine how many nodes to SWAP 
        part0_nodes = [n for n in self.graph.nodes.values() if n.partition == 0]
        part1_nodes = [n for n in self.graph.nodes.values() if n.partition == 1]

        # If the graph is not large enough clamp
        #assert len(part0_nodes) >= self.mutation_size 
        swap_count = min(len(part0_nodes), len(part1_nodes), self.mutation_size)

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






if __name__ == "__main__":
    # 1 Instantiate with desired parameters
    random_seed = utils.generate_random_seed()
    max_iteration_size = 250
    
    for m in [1, 2, 5, 7, 10, 15 ,20, 30, 50, 100]:
        ils = ILS(
            graph_filename="Graph500.txt", 
            max_iterations=max_iteration_size,    # 
            mutation_size=m,       # 
            random_seed=random_seed
        )
        
        # 2 Run ILS
        best_cut = ils.run_ils()
        
        # 3 Collect run statistics
        stats = ils.get_run_statistics()
        
        print(f"ILS best cut size for mutation size of :{m} found: {best_cut}")
        break
        
        
        
        
        
    #print("---- Summary ----")
    for k, v in stats.items():
        pass#print(f"{k}: {v}")





