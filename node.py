class Node:
    # a node class to be used in the graph
    
    def __init__(self, node_id: int, x_in : int = 0, y_in : int = 0):
        self.id = node_id
        self.neighbors = []        
        self.x = x_in
        self.y = y_in 
        
    
    def add_neighbor(self, neighbor_id: int):
        self.neighbors.append(neighbor_id)
   
    def __str__(self):
        return f"Node({self.id}, neighbors={self.neighbors})"
    
    def print(self):
        print(self.__str__())