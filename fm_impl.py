from graph import Graph
from node_linked import LinkedNode
import time

class FM:
    def __init__(self, graph:Graph):
        self.graph = graph   
        self.max_gain = -1        
        self.cut_size = 0     
        self.runs = 0
        self.run_durations = []
        self.__initialize_buckets()        
    
    def __initialize_buckets(self):                        
        #Find the max gain, we need it to initialize the buckets.
        self.max_gain = 0
        cut_size = 0
        for node_id in self.graph.nodes:
            linked_node = self.graph.nodes[node_id]
            gain,cut = self.graph.calculate_gain(linked_node)
            cut_size += cut
            # if gain > self.max_gain:
            #     self.max_gain = gain 
            if linked_node.get_degree() > self.max_gain:
                self.max_gain = linked_node.get_degree()
                
        #Initialize the buckets, assign -1 to denote nothing is in that gain index.
        self.cut_size = cut_size // 2
        max = self.max_gain
        
        #range: -max_gain to max_gain and there is 0.
        gains_left  = [None for _ in range(2 * max + 1)]
        gains_right = [None for _ in range(2 * max + 1)]
        parts = [gains_left, gains_right]        
        sizes = [0, 0]
        
        for node_id in self.graph.nodes:
            if max == 0:
                break #there is nothing to do. solution is already optimal.
            # '+ max' to shift the index to positive.
            linked_node = self.graph.nodes[node_id]
            gain = linked_node.last_calculated_gain + max             
            
            if parts[linked_node.partition][gain] is None:
                parts[linked_node.partition][gain] = linked_node                
            else:
                parts[linked_node.partition][gain].insert_after(linked_node)
            
            #update the size tracker. This is for not to calculate partition size every time for balancing.
            sizes[linked_node.partition] += 1 
            
        self.gains_left = gains_left
        self.gains_right = gains_right
        self.size_part_1 = sizes[0]
        self.size_part_2 = sizes[1]        
        pass
    
    def get_node_from_left_bucket(self, gain:int):
        if gain < -self.max_gain or gain > self.max_gain:
            raise ValueError(f"Invalid gain value. The gain value must be between -{self.max_gain} and max_gain {self.max_gain}.")
        return self.gains_left[gain + self.max_gain]
    
    def get_node_from_right_bucket(self, gain:int):
        if gain < -self.max_gain or gain > self.max_gain:
            raise ValueError(f"Invalid gain value. The gain value must be between -{self.max_gain} and max_gain {self.max_gain}.")
        return self.gains_right[gain + self.max_gain]
    
    def run_single_pass(self):        
        #TODO: if the sizes are equal, it must be random pick. But if we do that, the tests will fail.
        #We need to make this a flag like 'random_pick' to make it random.
        current_gain_list = self.gains_left 
        if self.graph.get_partition_balance() < 0:
            # If the left partition is smaller, pick one from the right partition.
            current_gain_list = self.gains_right 
        max = self.max_gain                
        
        #Search for the node with the highest gain.
        node = None
        for gain in range(2 * max, -1, -1):
            node = current_gain_list[gain]
            if node == None:
                continue            
            #Found the node with the highest gain.
            #assert gain == node.last_calculated_gain + max, f"Invalid gain value. Expected: {node.last_calculated_gain + max}, Actual: {gain}"
            break
        
        if node == None:
            #There is no node to move.
            return None        
        
        #Remove the node from the linked list and lock it to prevent it from being moved again.
        #The next node in the linked list will replace the removed node.
        current_gain_list[gain] = node.remove(locked=True)
        
        #Perform the move operation and update gains.
        return self.__move_node(node)
        
    def __move_node(self, node:LinkedNode):
        #Buffer the cut size before the pass.
        cut_size_before = self.cut_size
        max = self.max_gain    
        
        #Move the node to the other partition.
        self.graph.move_node(node.id)
        
        #Recalculate the gain of the node        
        gain,cut = self.graph.calculate_gain(node)
        #self.cut_size -= cut #TODO: this is BUG!!!
        parts = [self.gains_left, self.gains_right]
        
        #Update the gain buckets for the neighbors of the node.
        for neighbor_id in node.neighbors:
            neighbor = self.graph.nodes[neighbor_id]
            n_old = neighbor.last_calculated_gain
            n_new, cut = self.graph.calculate_gain(neighbor)
            #self.cut_size -= cut#TODO: this is BUG!!!
            
            #Check if the gain of the neighbor is changed and it is not locked.
            if n_old != n_new and not neighbor.locked:
                #The gain of the neighbor is changed, update the gain buckets.
                bucket = parts[neighbor.partition]
                #remove the neighbor from the old gain bucket.
                bucket[n_old + max] = neighbor.remove()
                
                # if n_new+max > 2*max or n_new+max < 0:
                #     print(f"Invalid gain value: {n_new}, {n_old}, {n_new+max}")
                #     pass
                #Insert the neighbor to the new gain bucket.
                if bucket[n_new + max] is None:
                    bucket[n_new + max] = neighbor
                else:
                    bucket[n_new + max].insert_after(neighbor)                
        
        #NOTE: if we can update the cut size as we move the nodes, we can avoid this array scan.
        self.cut_size = self.graph.get_cut_size()     
        return cut_size_before, self.cut_size, node.id, self.graph.is_balanced   

    def run_fm(self):
        #part1 = self.graph.get_partition(0)
        #part2 = self.graph.get_partition(1)
        solutions = []
            
        while True:        
            start_time = time.time()
            step_result = self.run_single_pass()
            end_time = time.time()
            self.run_durations.append(end_time - start_time)
            #print(f"Time taken for single pass: {end_time - start_time:.6f} seconds")
            self.runs += 1
            if step_result == None:
                break # Run until both buckets are exhausted.
            
            solutions.append(step_result)
            
        if len(solutions) == 0:
            return self.cut_size
        
        # Find the best cut, it has to be smalles cut size and it must be balanced.
        best_cut = None        
        for solution in solutions:
            _, cut_size_after, _, is_balanced = solution
            if is_balanced and (best_cut is None or cut_size_after < best_cut[1]):
                best_cut = solution                
        
        # We are doing hill climbing, so we need to revert the bad moves.
        # reverse loop the solutions until we find the best cut.
        for solution in reversed(solutions):
            if solution == best_cut:
                break # We found the best cut, stop reverting.
            
            #Revert the move until best cut.
            _, _, node_id, _ = solution
            self.graph.move_node(node_id)
        
        self.cut_size = self.graph.get_cut_size()
        self.graph.reset_and_free_nodes()
        return self.cut_size
        
    def get_run_statistics(self):
        return {
            "runs": self.runs,
            "run_times": self.run_durations,
            "total_elapsed": sum(self.run_durations),
            "average_elapsed": sum(self.run_durations) / len(self.run_durations)
        }