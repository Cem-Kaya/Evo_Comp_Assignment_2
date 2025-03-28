import random
from node_linked import LinkedNode
import utils

class Graph:
    # a class for the graph
    
    def __init__(self, filename: str):
        self.nodes = {}  # Dictionary {node_id: Node}
        self.load_graph(filename)
        #self.solution = {}  # Dictionary {node_id: partition_id}
        self.is_balanced = False
        self.size_tracker = [0,0]
        self.cut_size = -1 
        
    def load_graph(self, filename: str):
        #Reads the graph from a file  
        with open(filename, 'r') as f:
            for line in f:
                parts = line.split()
                node_id = int(parts[0])
                x, y = map(float, parts[1][1:-1].split(","))  # Extract coordinates from (x,y)

                if node_id not in self.nodes:
                    self.nodes[node_id] = LinkedNode(node_id, x, y)
                else:
                    self.nodes[node_id].x, self.nodes[node_id].y = x, y  # Update 

                # neighbors
                for neighbor_id in parts[3:]:
                    neighbor_id = int(neighbor_id)

                    if neighbor_id not in self.nodes:
                        self.nodes[neighbor_id] = LinkedNode(neighbor_id)  # No coordinates available

                    self.nodes[node_id].add_neighbor(neighbor_id)
    
    def get_cut_size(self):
        #Computes the number of edges that cross partitions
        cut_size = 0
        for node in self.nodes.values():
            for neighbor_id in node.neighbors:
                if node.partition != self.nodes[neighbor_id].partition:
                    cut_size += 1
              
        # Each edge counted twice      
        self.cut_size = cut_size // 2 # update the cut size. This is useful for debugging and ToString method.
        return self.cut_size 

    def set_random_solution(self, seed=None):
        #Generates a random balanced partitioning of nodes
        node_ids = list(self.nodes.keys())  
        
        if seed is not None:
            seed = utils.generate_random_seed()
            random.seed(seed) 
        random.shuffle(node_ids)   
        
        half_len= len(node_ids) // 2  # Half of the total nodes
        new_solution = {} 
        # Assign half of the nodes to partition 0 and the rest to partition 1
        for i in range(half_len):
            new_solution[node_ids[i]] = 0
        for i in range(half_len , len(node_ids)):
            new_solution[node_ids[i]] = 1
            
        #This will update the partition of the nodes and cut size.
        self.set_solution_explicit(new_solution) 
    
    def reset_and_free_nodes(self):
        """This method will remove the next and prev pointers from linked nodes and set the locked=False.
        It will also reset the last_calculated_gain and last_calculated_cut values.
        """
        #Free the nodes
        for node in self.nodes.values():
            node.free()
            node.reset()
            node.last_calculated_gain = -1
            node.last_calculated_cut = -1
    
    def set_solution_explicit(self, solution: dict):
        assert len(solution)% 2 == 0, "The number of nodes must be even."
        self.size_tracker = [0,0]
        
        #reset the nodes
        self.reset_and_free_nodes()
        
        #Update the partition of the nodes
        for node_id in solution:
            self.nodes[node_id].partition = solution[node_id]
            self.size_tracker[solution[node_id]] += 1           
        
        self.is_balanced = self.size_tracker[0] == self.size_tracker[1]
        assert self.is_balanced, "The solution is not balanced."        
        self.get_cut_size() #This will update the cut size property.
        
    def move_node(self, node_id: int):
        dest= 1- self.nodes[node_id].partition
        self.nodes[node_id].partition = dest
        self.size_tracker[dest] += 1
        self.size_tracker[1 - dest] -= 1
        self.is_balanced = self.size_tracker[0] == self.size_tracker[1]
    
    def get_partition_balance(self) -> int:
        """Gets the balance of the partition. Returns an integer value indicating the balance.
        
        Returns:
            int: less than 0 if partition 0 is smaller, greater than 0 if partition 1 is smaller, 0 if equal.
        """
        return self.size_tracker[0] - self.size_tracker[1]
    
    def calculate_gain(self, node: LinkedNode):
        #Calculate the gain of the node. If a neighbor is in the same partition, gain is decreased by 1,
        gain = 0
        cut = 0
        for neighbor_id in node.neighbors:            
            if node.partition != self.nodes[neighbor_id].partition:
                gain += 1
                cut += 1
            else:
                gain -= 1
                
        node.last_calculated_gain = gain
        node.last_calculated_cut = cut
        return gain,cut
    
    def get_partition(self, partition:int):
        return [node.id for node in self.nodes.values() if node.partition == partition] 
            
    def __str__(self):
        string_ver = ""
        # stringfy the extra data 
        string_ver+= f"cut_size = {self.cut_size} \n"
        string_ver+= f"is_balanced = {self.is_balanced} \n"
        string_ver+= f"unbalanced_type = {self.unbalanced_type} \n"
        string_ver+= f"number of nodes = {len(self.nodes)} \n"
        string_ver+= f"number of edges = {self.get_cut_size()} \n"
        string_ver+= "Solution: \n"
        string_ver+= str(self.solution) + "\n"
        string_ver+= "Nodes: \n"
        # stringfy the graph        
        for node in self.nodes.values():
            string_ver += str(node) + "\n"
        return string_ver

