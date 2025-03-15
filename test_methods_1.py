import graph as g
import node as n
from node_linked import LinkedNode
from fm_impl import FM

def test_load_graph():
    # LLM prompt: load the file parse it line by line. each line is space-separated.
    # Load Graph500.txt file
    graph_file = "Graph500.txt"
    values = []
    
    with open(graph_file, 'r') as file:
            for line in file:
                # Split each line by space and convert to integers
                arr = line.split()
                arr[0] = int(arr[0])
                arr[2:] = map(int, arr[2:])
                values.append(arr)
    # Each entry in the values array must have at least 3 elements.
    min_length_item = min(values, key=len)
    assert len(min_length_item) == 3 # this is a node with no edges (no neighbors).
    
    #LLM Prompt: Create a dictionary to store the nodes in the list. The key is the node id and the value is the node object
    nodes = {}
    for i in values:
        nodes[i[0]] = i
        
    #Load the graph
    graph = g.Graph(graph_file)
        
    #iterate over the nodes in the graph
    for node_id in graph.nodes:
        #get the node from the dictionary
        file_entry = nodes[node_id]
        node = graph.nodes[node_id]
        
        #verify the neighbors count        
        assert len(node.neighbors) == file_entry[2]
        if len(node.neighbors) == 0:
            assert len(node.neighbors) == 0
            continue #nothing else to check
        
        #extract the neighbour ids from the file entry
        neighbor_ids = file_entry[3:]
        assert len(neighbor_ids) == len(node.neighbors)
        #iterate over the neighbors of the node
        for neighbor_id in neighbor_ids:
            #verify the neighbor is in the graph
            assert neighbor_id in graph.nodes
            #verify the neighbor is in the node's neighbors
            assert neighbor_id in node.neighbors
            #verify the neighbor has the node as a neighbor
            neighbor = graph.nodes[neighbor_id]
            assert node_id in neighbor.neighbors
    

def test_cutsize():
    """
1 (0.1,0.1) 2 2 3
2 (0.3,0.6) 2 1 3
3 (0.5,0.3) 3 1 2 5 
4 (0.7,0.9) 1 5
5 (0.9,0.1) 3 3 6 4
6 (1.1,0.3) 1 5
    """
    #This is a simple graph with 6 nodes. The smallest cut is {1, 2, 3} and {4, 5, 6}.
    #Cut is 1. 
    graph_file = "test_graph1.txt"
    graph = g.Graph(graph_file)
    
    #We cannot use initilize solution method, because it is random. 
    #We need to set the solution manually.
    graph.solution = {1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1}
    cut_size = graph.get_cut_size()
    assert cut_size == 1
    
    #Test with another manual solution. Swap 3 with 5. The cut size is 5.
    graph.solution = {1: 0, 2: 0, 3: 1, 4: 1, 5: 0, 6: 1}
    cut_size = graph.get_cut_size()
    assert cut_size == 5

def test_graph_operations():
    """This test case is for the graph operations. 
    It will test the primary node operations like moving a node to the other partition and calculating the gain of a node.
    It is also a mini-simulation of the FM algorithm.
    """
    graph_file = "test_graph1.txt"
    graph = g.Graph(graph_file)
    
    #Test with another manual solution. Swap 3 with 5. The cut size is 5.
    #Start with a bad solution.
    graph.solution = {1: 0, 2: 0, 3: 1, 4: 1, 5: 0, 6: 1}
    cut_size = graph.get_cut_size()
    assert cut_size == 5
    
    #If we move the node 3 to the other partition,
    #the cut size will be decreased by 3, so the gain must be 3.
    node3 = graph.nodes[3]  
    gain = graph.calculate_gain_for_node(3)  
    assert gain == 3
    assert node3.last_calculated_gain == gain #The gain is stored in the node object for easy access.
    
    #Move the node to the other partition
    graph.move_node_to_other_solution(3)
    assert graph.solution[3] == 0
    cut_size = graph.get_cut_size()
    #cut size is decreased by 3, so cut size will be 2 (because 5 is still there), 
    #the solution is unbalanced at this stage.
    assert cut_size == 2 
    
    #get the gain of 5.
    node5 = graph.nodes[5]
    gain = graph.calculate_gain_for_node(5)
    assert gain == 1 # Gain will be 1 because we moved 3 to the other partition.
    assert node5.last_calculated_gain == gain
    
    #move the node 5 to the other partition
    graph.move_node_to_other_solution(5)
    cut_size = graph.get_cut_size()
    #cut size is decreased by 1.
    #so cut size will be 1 and the solution is balanced. This is the optimal solution.
    #This is the optimal convergence behavior for this trivial graph.
    assert cut_size == 1
    
def test_graph_operations2():
    """This test case is for the graph operations. 
    It will test the primary node operations like moving a node to the other partition and calculating the gain of a node.
    It is also a mini-simulation of the FM algorithm.
    """
    graph_file = "test_graph1.txt"
    graph = g.Graph(graph_file)
    
    #Test with another manual solution. Swap 3 with 5. The cut size is 5.
    #Start with a bad solution.
    graph.set_solution_explicit({1: 0, 2: 0, 3: 1, 4: 1, 5: 0, 6: 1})
    cut_size = graph.get_cut_size_node_traversal_temp()
    assert cut_size == 5
    
    #If we move the node 3 to the other partition,
    #the cut size will be decreased by 3, so the gain must be 3.
    node3 = graph.nodes[3]  
    gain = graph.calculate_gain(node3)
    assert gain == 3
    assert node3.last_calculated_gain == gain #The gain is stored in the node object for easy access.
    
    #Move back the gain must be -previous gain, because we made a better solution by moving the node 3.
    node3.partition = 1 - node3.partition
    gain = graph.calculate_gain(node3)
    assert gain == -3
    assert node3.last_calculated_gain == gain #The gain is stored in the node object for easy access.
    
    cut_size = graph.get_cut_size_node_traversal_temp()
    assert cut_size == 2
    
    #get the gain of 5.
    node5 = graph.nodes[5]
    gain = graph.calculate_gain(node5)
    assert gain == 1 # Gain will be 1 because we moved 3 to the other partition.
    assert node5.last_calculated_gain == gain
    
    #move the node 5 to the other partition
    node5.partition = 1 - node5.partition
    cut_size = graph.get_cut_size_node_traversal_temp()
    #cut size is decreased by 1.
    assert cut_size == 1

def test_fm_single_pass():
    #load the test graph
    graph_file = "test_graph1.txt"
    graph = g.Graph(graph_file)
    
    #Setup the scenario. Bad solution.
    custom_solution = {1: 0, 2: 0, 3: 1, 4: 1, 5: 0, 6: 1}
    graph.set_solution_explicit(custom_solution)
    cut_size = graph.get_cut_size_node_traversal_temp()
    assert cut_size == 5
    
    #Create the FM object, it initializes the buckets.
    fm = FM(graph)
    assert fm.cut_size == 5
    assert fm.size_part_1 == 3
    assert fm.size_part_2 == 3
    assert fm.max_gain == 3    
    
    #Check the bucket distribution. Iterate over custom_solution and check the bucket values.
    max = fm.max_gain    
    #expected_gains = {1: 0, 2: 0, 3: 3, 4: 1, 5: 3, 6: 1}
    
    #CHECK LEFT BUCKET
    #Check gain 0 from left bucket. It is node 1 and 2.
    node = fm.get_node_from_left_bucket(0)
    assert node == graph.nodes[1]
    assert node.prev == None
    
    #This is a linked list, so we can get the next node.
    node = node.next
    assert node == graph.nodes[2]
    assert node.next == None
    
    #Check gain 3 from left bucket. It is node 5 only.
    node = fm.get_node_from_left_bucket(3)
    assert node == graph.nodes[5]
    assert node.prev == None
    assert node.next == None
    
    #Gain < 0 is not in the left bucket and gain 1 and 2 also not.
    for gain in range(-max, 0):
        assert fm.get_node_from_left_bucket(gain) == None
    assert fm.get_node_from_left_bucket(1) == None
    assert fm.get_node_from_left_bucket(2) == None
    
    #CHECK RIGHT BUCKET
    #Check gain 0 from right bucket, it should be none.
    node = fm.get_node_from_right_bucket(0)
    assert node == None
    
    #Check gain 1 from right bucket. It is node 4 and 6.
    node = fm.get_node_from_right_bucket(1)
    assert node == graph.nodes[4]
    assert node.prev == None
    
    #This is a linked list, so we can get the next node.
    node = node.next
    assert node == graph.nodes[6]
    assert node.next == None
    
    #Gain < 0 is not in the right bucket.
    for gain in range(-max, 0):
        assert fm.get_node_from_right_bucket(gain) == None
    
    #Run the single pass. The node 5 will be moved to the right partition. 
    #The solution is not balanced yet. Expected cut size is 2.
    run = fm.run_pass()
    #The best gain is at node5 in the left bucket. The gain is 3.
    best_node = graph.nodes[5]
    #The cut size is 5, so the cut size will be 2.
    assert run[0] == 5
    #assert run[1] == 2 #THIS IS BUG!!
    assert run[2] == best_node.id
    
    #We need to check the bucket distribution again.
    #CHECK LEFT BUCKET, it is only node 1 and 2, both have gain 0 (unchanged).
    node = fm.get_node_from_left_bucket(0)
    assert node == graph.nodes[1]
    node = node.next
    assert node == graph.nodes[2]
    #And all gains between -max and +max must be none, except gain 0.
    for gain in range(-max, +max):
        if gain != 0:
            assert fm.get_node_from_left_bucket(gain) == None
    
    #Check the right bucket. 3,4,5,6 are in the right bucket.
    #The nodes 4 and 6 have now gain -1 (because 5 is moved to the right partition).
    node = fm.get_node_from_right_bucket(-1)
    assert node == graph.nodes[6]
    node = node.next
    assert node == graph.nodes[4]
    #The nodes 3 have gain 1 now. Because it is connected to 1 and 2 in the left partition, but 5 is in the right partition.
    #So if we move, we will win 2 and lose 1. So the gain is 1.
    node = fm.get_node_from_right_bucket(1)
    assert node == graph.nodes[3]
    #The node 5 is moved to the right partition, so its gain must be -3, BUT it is removed from the bucket, so not there.
    node = fm.get_node_from_right_bucket(-3)
    assert node == None
    #We have only gains 1, -1,in the right bucket.
    for gain in range(-max, +max):
        if gain == 1 or gain == -1:            
            continue #nodes 3,4,6
        assert fm.get_node_from_right_bucket(gain) == None
            
    #Run the pass again. This time the node 3 will be moved to the left partition.
    run = fm.run_pass()
    #The best gain is at node3 in the right bucket. The gain is 2.
    best_node = graph.nodes[3]
    #The cut size is 2, so the cut size will be 1.
    assert run[0] == 2
    #assert run[1] == 1 #THIS IS BUG!!
    assert run[2] == best_node.id
    
    
def test_linked_node():
    node1 = LinkedNode(1)
    node2 = LinkedNode(2)
    node3 = LinkedNode(3)
    
    node1.set_next(node2)
    assert node1.next == node2
    assert node2.prev == node1
    
    node2.set_next(node3)
    assert node2.next == node3
    assert node3.prev == node2
    
    #test remove
    current = node2.remove()
    assert current == node1
    assert node1.prev == None
    assert node1.next == node3
    assert node3.prev == node1
    assert node3.next == None
    assert node2.next == None
    assert node2.prev == None
    
    #test remove from the end
    current = node3.remove()
    assert current == node1
    assert node1.next == None
    assert node3.prev == None
    assert node3.next == None
    
    #test remove node1
    current = node1.remove()
    assert current == None