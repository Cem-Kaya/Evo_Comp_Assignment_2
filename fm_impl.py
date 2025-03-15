from graph import Graph

class FM:
    def __init__(self, graph:Graph):
        self.graph = graph   
        self.max_gain = -1
        self.size_part_1 = 0
        self.size_part_2 = 0
        self.cut_size = 0     
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
    
    def run_pass(self):
        #Buffer the cut size before the pass.
        cut_size_before = self.cut_size
        
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
            break
        
        #Remove the node from the linked list, and assign the next node to the current gain index.
        current_gain_list[gain] = node.remove()
        #Update the size tracker.
        if current_gain_list == self.gains_left:
            self.size_part_1 -= 1
        else:
            self.size_part_2 -= 1
        
        #Move the node to the other partition.
        node.partition = 1 - node.partition
        #Recalculate the gain of the node and update the cut size.
        gain,cut = self.graph.calculate_gain(node)
        self.cut_size -= cut #TODO: this is BUG!!!
        parts = [self.gains_left, self.gains_right]
        
        #Update the gain buckets for the neighbors of the node.
        for neighbor_id in node.neighbors:
            neighbor = self.graph.nodes[neighbor_id]
            n_old = neighbor.last_calculated_gain
            n_new, cut = self.graph.calculate_gain(neighbor)
            self.cut_size -= cut#TODO: this is BUG!!!
            
            #Check if the gain of the neighbor is changed.
            if n_old != neighbor.last_calculated_gain:
                #The gain of the neighbor is changed, update the gain buckets.
                bucket = parts[neighbor.partition]
                #remove the neighbor from the old gain bucket.
                bucket[n_old + max] = neighbor.remove()
                
                #Insert the neighbor to the new gain bucket.
                if bucket[n_new + max] is None:
                    bucket[n_new + max] = neighbor
                else:
                    bucket[n_new + max].insert_after(neighbor)
                    
        #NOTE: if the cut size is not improved, we can revert the move.
        #We can do this by moving the node to the other partition again.
        #We need to update the bucket lists and the cut size back, but we won't put the current node
        #to the gain bucket again. we can return the old cut size, new_cut_size and the current node.        
        return cut_size_before, self.cut_size, node.id
        
        