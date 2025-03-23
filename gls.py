import random
import time
import statistics
from fm_impl import FM
from graph import Graph

from multiprocessing import Process, Queue


class GLS:
    """
    Genetic Local Search + FM local heuristic 

    pop_size: size of the population
    max_iterations: number of generations
    For each generation:
        1 Pair up parents 
        2 Produce children via crossover  ( what type ? ? ?  TODO play with other ttypes beside uniform )
        3 Optimize each child via FM
        4 Combine old population + children -> select top pop_size (TODO maybe add some randomness here )
        5 Keep track of best solution found so far.
    """

    def __init__(self, graph_filename: str, pop_size=50, max_iterations=250, random_seed=None):
        self.graph_filename = graph_filename
        self.pop_size = pop_size
        
        self.max_iterations = max_iterations
        
        self.random_seed = random_seed
        if self.random_seed is not None:
            random.seed(self.random_seed)

        # Each element in population is a dict:
        #[
        #   {
        #      "solution": { node_id: 0 , ... },
        #      "cut_size": <integer>
        #    }, ...
        # ]
        self.population = []

        self.best_cut_size = None
        self.best_solution = None
        self.start_time = None
        self.end_time = None

        self.iteration_data = []  # to store iteration stats

    def select_top(self, combined ):
        #sort
        combined.sort(key=lambda x: x["cut_size"])  
        #select upper half 
        return combined[:self.pop_size]

    def run_gls(self):
        """        
          1 Initialize population 
          2 Repeat for max_iterations:
              a Shuffle pair up parents.
              b For each pair -> produce one child -> optimize with FM.
              c Combine old population + new children; keep only top pop_size.
              d Track best solution found so far.
        """
        self.start_time = time.time()
        self._init_population()

        # Main generational loop
        for iteration in range(1, self.max_iterations + 1):

            #  Shuffle -> pair up parents
            random.shuffle(self.population)

            #  For each pair -> produce a child -> optimize
            new_children = []
            
            assert self.pop_size % 2 == 0, "Population size must be even for pairing."
            for i in range(0, self.pop_size, 2):
                parent1 = self.population[i]
                parent2 = self.population[i+1]

                # produce child by crossover
                child_solution = self._uniform_crossover(parent1["solution"], parent2["solution"])
                #FM optimization
                child_cut_size = self._optimize_with_fm(child_solution)

                new_children.append({
                    "solution": child_solution,
                    "cut_size": child_cut_size
                })

            #  Combine old pop + new children and select the top pop_size
            combined = self.population + new_children
            #selection
            selected = self.select_top(combined)
             
            self.population = selected  

            # best solution 
            if self.population[0]["cut_size"] < self.best_cut_size:
                self.best_cut_size = self.population[0]["cut_size"]
                self.best_solution = dict(self.population[0]["solution"])

            self.iteration_data.append({
                "iteration": iteration,
                "best_cut_size_so_far": self.best_cut_size,
                "best_cut_size_this_iter": min(child["cut_size"] for child in new_children)
            })

        self.end_time = time.time()
        return self.best_cut_size

    def get_run_statistics(self):        
        # get a dict summarizing the run
        
        total_time = (self.end_time - self.start_time) if (self.start_time and self.end_time) else 0.0
        best_by_iteration = [it["best_cut_size_so_far"] for it in self.iteration_data]
        if not best_by_iteration:
            best_by_iteration = [self.best_cut_size] if self.best_cut_size else []

        return {
            "pop_size": self.pop_size,
            "max_iterations": self.max_iterations,
            "best_cut_size": self.best_cut_size,
            "time_elapsed": total_time,
            "average_best_by_iteration": statistics.mean(best_by_iteration),
            "iteration_log": self.iteration_data
        }

    
    def _init_population(self):
        
        self.population.clear()
        while len(self.population) < self.pop_size:
            # Build a random balanced solution
            random_solution = self._create_random_balanced_solution()

            # Optimize it with FM
            cut_size = self._optimize_with_fm(random_solution)

            self.population.append({
                "solution": random_solution,
                "cut_size": cut_size
            })

        # Track best in population
        self.population.sort(key=lambda x: x["cut_size"])
        self.best_cut_size = self.population[0]["cut_size"]
        self.best_solution = dict(self.population[0]["solution"])

    def _optimize_with_fm(self, solution_dict):
        graph = Graph(self.graph_filename)
        graph.set_solution_explicit(solution_dict)

        fm_impl = FM(graph)
        fm_impl.run_fm()

        final_cut = graph.get_cut_size()

        # Update the original dict so external references see the improved partition
        for nid, node in graph.nodes.items():
            solution_dict[nid] = node.partition

        return final_cut

    def _create_random_balanced_solution(self):
        
        tmp_graph = Graph(self.graph_filename)
        tmp_graph.set_random_solution()
        return {nid: node.partition for nid, node in tmp_graph.nodes.items()}

    def _uniform_crossover(self, parentA, parentB):
        # TODO make it balanced 
        all_nodes = sorted(parentA.keys())
        num_nodes = len(all_nodes)
        half = num_nodes // 2  # number of  1

        child = {}
        assigned_ones = 0

        # Assign bits where parents agree
        for nid in all_nodes:
            if parentA[nid] == parentB[nid]:
                bit = parentA[nid]
                child[nid] = bit
                assigned_ones += bit
            else:
                # Mark differing nodes as None
                child[nid] = None

        # 2 get how many  1 bits missing
        needed_ones = half - assigned_ones

        # get the nodes that are still None
        differing_nodes = [nid for nid, bit in child.items() if bit is None]

        # 3 Assign bits for the differing_nodes to hit the exact total of  ones required
        if needed_ones > 0:# if we need more ones            
            chosen_for_ones = random.sample(differing_nodes, needed_ones)
            for nid in differing_nodes:
                child[nid] = 1 if nid in chosen_for_ones else 0
        elif needed_ones < 0:# We have too many ones already, so we need to convert some to zero
            
            must_be_zero = random.sample(differing_nodes, abs(needed_ones))
            for nid in differing_nodes:
                child[nid] = 0 if nid in must_be_zero else 1
        else:            
            # so half of these differing bits become 1, the other half 0
            half_diff = len(differing_nodes) // 2
            chosen_for_ones = random.sample(differing_nodes, half_diff)
            for nid in differing_nodes:
                child[nid] = 1 if nid in chosen_for_ones else 0

        return child


if __name__ == "__main__":
  
    pop_sizes_to_test = [4, 10, 20, 50, 100,200, 500] # must be even 

    for pop_size in pop_sizes_to_test:
        gls = GLS(
            graph_filename="Graph500.txt",
            pop_size=pop_size,
            max_iterations=250,
            random_seed=42
        )
        best_cut = gls.run_gls()
        stats = gls.get_run_statistics()

        print(f"GLS best cut size for pop_size={pop_size}: {best_cut}")
      
