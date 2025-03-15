class Node:
    # a node class to be used in the graph
    
    def __init__(self, node_id: int, x_in : int = 0, y_in : int = 0):
        self.id = node_id
        self.neighbors = []        
        self.x = x_in
        self.y = y_in         
        self.locked = False
        self.last_calculated_gain = -1
        self.partition = -1
        
    def lock(self):
        self.locked = True
        
    def free(self):
        self.locked = False
    
    def add_neighbor(self, neighbor_id: int):
        self.neighbors.append(neighbor_id)
        
    def get_degree(self):
        return len(self.neighbors)
        
    def __str__(self):
        return f"Node({self.id}, neighbors={self.neighbors})"
    
    def print(self):
        print(self.__str__())