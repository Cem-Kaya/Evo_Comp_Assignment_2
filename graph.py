import random
from node_linked import LinkedNode

class Graph:
    # a class for the graph
    
    def __init__(self, filename: str):
        self.nodes = {}  # Dictionary {node_id: Node}
        self.load_graph(filename)
        self.solution = {}  # Dictionary {node_id: partition_id}
        self.is_balanced = False
        self.unbalanced_type = "unknown" # extra0 or extra1
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
                    #13.03.2025 MS: The line below duplicates the neighbor array of current node. 
                    #Say, current node is node 1 and it has a neighbor 2. 
                    #The line below will add 1 to the neighbor array of node 2.
                    #When the current node is 2, the line below will add 2 to the neighbor array of node 1.
                    #So the 'node 1' will have 'node 2' twice in the neighbor array. 
                    #We don't need this line, because we assume that given graph dataset is correct.
                    #If the dataset is correct, each node will update its neighbors array and the graph will be undirected.
                    #There is a test case to check if the graph is undirected.
                    #self.nodes[neighbor_id].add_neighbor(node_id)  # Undirected graph


    def get_cut_size(self):
        #Computes the number of edges that cross partitions
        cut_size = 0
        for node in self.nodes.values():
            for neighbor_id in node.neighbors:
                if self.solution[node.id] != self.solution[neighbor_id]:
                    cut_size += 1
                    
        self.cut_size = cut_size #update the cut size. This is useful for debugging and ToString method.
        return cut_size // 2  # Each edge counted twice
    
    def get_cut_size_node_traversal_temp(self):
        #Computes the number of edges that cross partitions
        cut_size = 0
        for node in self.nodes.values():
            for neighbor_id in node.neighbors:
                if node.partition != self.nodes[neighbor_id].partition:
                    cut_size += 1
                    
        self.cut_size = cut_size #update the cut size. This is useful for debugging and ToString method.
        return cut_size // 2  # Each edge counted twice

    def set_random_solution(self, seed=None):
        #reset the nodes
        for node in self.nodes.values():
            node.reset()
        #Generates a random balanced partitioning of nodes
        node_ids = list(self.nodes.keys())  
        
        if seed is not None:
            random.seed(seed) 
        random.shuffle(node_ids )   
        
        half_len= len(node_ids) // 2  # Half of the total nodes
        new_solution = {} 
        # Assign half of the nodes to partition 0 and the rest to partition 1
        for i in range(half_len):
            new_solution[node_ids[i]] = 0
        for i in range(half_len , len(node_ids)):
            new_solution[node_ids[i]] = 1
            
        #This will update the partition of the nodes and cut size.
        self.set_solution_explicit(new_solution) 
    
    def set_solution_explicit(self, solution: dict):
        #Sets the solution explicitly
        self.solution = solution
        
        #Update the partition of the nodes
        for node_id in self.solution:
            self.nodes[node_id].partition = self.solution[node_id]
        
        assert len(self.solution)% 2 == 0, "The number of nodes must be even."
        self.is_balanced = True
        self.get_cut_size() #This will update the cut size property.        
        
    def move_node_to_other_solution(self, node_id:int):
        #switch the partition of the node
        self.solution[node_id] = 1 - self.solution[node_id]
        
    def calculate_gain_for_node(self, node_id:int):
        #Calculate the gain of the node. If a neighbor is in the same partition, gain is decreased by 1,
        #Because when we move this node to the other partition, the cut size will be increased by 1.
        gain = 0
        node = self.nodes[node_id]
        
        for neighbor_id in node.neighbors:
            if self.solution[node_id] != self.solution[neighbor_id]:
                gain += 1
            else:
                gain -= 1
                
        node.last_calculated_gain = gain
        return gain
    
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
        return gain,cut
            
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
    
    def mutate(self, mutation_size):
        pass 

