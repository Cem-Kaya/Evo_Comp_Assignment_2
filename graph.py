import random
from node import Node

class Graph:
    # a class for the graph
    
    def __init__(self, filename: str):
        self.nodes = {}  # Dictionary {node_id: Node}
        self.load_graph(filename)
        self.init_solution()
        self.solution = {}  # Dictionary {node_id: partition_id}
        
    def load_graph(self, filename: str):
        #Reads the graph from a file  
        with open(filename, 'r') as f:
            for line in f:
                parts = line.split()
                node_id = int(parts[0])
                x, y = map(float, parts[1][1:-1].split(","))  # Extract coordinates from (x,y)

                if node_id not in self.nodes:
                    self.nodes[node_id] = Node(node_id, x, y)
                else:
                    self.nodes[node_id].x, self.nodes[node_id].y = x, y  # Update 

                # neighbors
                for neighbor_id in parts[3:]:
                    neighbor_id = int(neighbor_id)

                    if neighbor_id not in self.nodes:
                        self.nodes[neighbor_id] = Node(neighbor_id)  # No coordinates available

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


    def init_solution(self):
        #Generates a ~balanced random partitioning
        node_list = list(self.nodes.keys())
        random.shuffle(node_list)
        self.solution = {node: 0 if i < len(node_list) // 2 else 1 for i, node in enumerate(node_list)}

    def get_cut_size(self):
        #Computes the number of edges that cross partitions
        cut_size = 0
        for node in self.nodes.values():
            for neighbor_id in node.neighbors:
                if self.solution[node.id] != self.solution[neighbor_id]:
                    cut_size += 1
        return cut_size // 2  # Each edge counted twice

    def mutate(self, mutation_size):
        pass 
