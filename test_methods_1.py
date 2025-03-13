import graph as g
import node as n

def test_graph():
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
    
    #create a dictionary to store the nodes in the list. The key is the node id and the value is the node object
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
    

test_graph()