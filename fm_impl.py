from graph import Graph
from node_linked import LinkedNode
import time

class FM:
    def __init__(self, graph:Graph):
        self.graph = graph   
        self.max_gain = -1
        self.size_part_1 = 0
        self.size_part_2 = 0
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
            if gain > self.max_gain:
                self.max_gain = gain 
                
        #Initialize the buckets, assign -1 to denote nothing is in that gain index.
        self.cut_size = cut_size // 2
        max = self.max_gain
        
        #range: -max_gain to max_gain and there is 0.
        gains_left  = [None for _ in range(2 * max + 1)]
        gains_right = [None for _ in range(2 * max + 1)]
        parts = [gains_left, gains_right]        
        sizes = [0, 0]
        
        for node_id in self.graph.nodes:
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
        current_gain_list = self.gains_left#TODO: if the sizes are equal, it must be random pick.
        if self.size_part_1 < self.size_part_2:
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
        
        #Remove the node from the linked list, and assign the next node to the current gain index.
        
        gain = current_gain_list.index(node)
        #Remove the node from the linked list, and assign the next node to the current gain index.
        #Lock the node to prevent it from being moved again.
        current_gain_list[gain] = node.remove(locked=True)
        #Update the size tracker.
        if current_gain_list == self.gains_left:
            self.size_part_1 -= 1
        else:
            self.size_part_2 -= 1
            
        return self.__move_node(node)
        
    def __move_node(self, node:LinkedNode):
        #Buffer the cut size before the pass.
        cut_size_before = self.cut_size
        max = self.max_gain    
        
        #Move the node to the other partition.
        node.partition = 1 - node.partition
        
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
                
                #Insert the neighbor to the new gain bucket.
                if bucket[n_new + max] is None:
                    bucket[n_new + max] = neighbor
                else:
                    bucket[n_new + max].insert_after(neighbor)                
        
        #NOTE: if we can update the cut size as we move the nodes, we can avoid this array scan.
        self.cut_size = self.graph.get_cut_size_node_traversal_temp()     
        return cut_size_before, self.cut_size, node.id    

    def run_fm(self):
        step_result = 0 # just initialize with some value    
        while step_result is not None:        
            start_time = time.time()
            step_result = self.run_single_pass()
            end_time = time.time()
            self.run_durations.append(end_time - start_time)
            #print(f"Time taken for single pass: {end_time - start_time:.6f} seconds")
            self.runs += 1
            if step_result is not None:
                cut_size_before, cut_size_after, node_id = step_result
                #print(f"Moved node {node_id} from partition 1 to 2. Cut size before: {cut_size_before}, after: {cut_size_after}")
                if cut_size_before < cut_size_after:
                    #Current cut is worse than before, revert the move.
                    node_to_revert = self.graph.nodes[node_id]                
                    self.__move_node(node_to_revert)
                    #print(f"Reverted the move. Moved node {node_id} from partition 2 to 1.")
                    
        return self.cut_size
        
        