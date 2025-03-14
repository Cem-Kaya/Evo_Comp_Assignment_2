from collections import defaultdict
import random

from graph import Graph
from node import Node



class FM:
    def __init__(self, graph):
        self.graph = graph
        self.left_bucket = defaultdict(list) # {5: {node_ids}, 4: {node_ids}} the first degree keys are the cut value of the node 
        self.right_bucket = defaultdict(list)# labeled 1 
        self.history = []
        for node in self.graph.nodes.values():             
            cut_effect = self.graph.get_cuts_per_node(node.id)
            if self.graph.solution[node.id] == 0:
                self.left_bucket[cut_effect].extend([node.id])
            else: # 1 
                self.right_bucket[cut_effect].extend([node.id])
        
        assert graph.is_balanced
        assert graph.cut_size != -1 
                
    def fm_iter (self):
        # chose node if unbalanced get from the higher side else get the highest gain node
        selected_node = None
        node_grain = -1 
        if self.graph.is_balanced:
            #select the bucket with the highest cut value 
            left_max = max(self.left_bucket.keys())
            right_max = max(self.right_bucket.keys())
            if left_max > right_max:
                rand_idx = random.randrange(len(self.left_bucket[left_max]))  # Get a random index
                selected_node = self.left_bucket.pop(rand_idx) 
                self.graph.is_balanced = False
                self.unbalanced_type = "extra0"
                node_grain = left_max
            else:
                rand_idx = random.randrange(len(self.right_bucket[right_max]))  # Get a random index
                selected_node = self.right_bucket.pop(rand_idx) 
                self.graph.is_balanced = False
                self.unbalanced_type = "extra1"
                node_grain = right_max
        else:
            if self.unbalanced_type == "extra0":
                left_max = max(self.left_bucket.keys())
                rand_idx = random.randrange(len(self.left_bucket[left_max]))
                selected_node = self.left_bucket.pop(rand_idx) 
                node_grain = left_max
            else: 
                right_max = max(self.right_bucket.keys())
                rand_idx = random.randrange(len(self.right_bucket[right_max]))
                selected_node = self.right_bucket.pop(rand_idx) 
                node_grain = right_max
            graph.is_balanced = True # the diff should be 1 at max ! 
        # update the solution
        self.graph.solution[selected_node] = 1 - self.graph.solution[selected_node]   
        self.graph.cut_size = self.graph.cut_size - node_grain + (self.graph.nodes[selected_node].neighbors) - node_grain
        # update connected nodes 
            
            
                
if __name__ == "__main__":
    graph = Graph("my_mini_graph.txt")
    graph.set_random_solution(0)
    fm = FM(graph)
    print(fm.left_bucket)
    print(fm.right_bucket)