import graph as g
import node as n
from node_linked import LinkedNode
from fm_impl import FM
import utils
import pandas as pd
from scipy import stats

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
    solution = {1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1}
    graph.set_solution_explicit(solution)
    cut_size = graph.get_cut_size()
    assert cut_size == 1
    
    #Test with another manual solution. Swap 3 with 5. The cut size is 5.
    solution = {1: 0, 2: 0, 3: 1, 4: 1, 5: 0, 6: 1}
    graph.set_solution_explicit(solution)
    cut_size = graph.get_cut_size()
    assert cut_size == 5
    """This test case is for the graph operations. 
    It will test the primary node operations like moving a node to the other partition and calculating the gain of a node.
    It is also a mini-simulation of the FM algorithm.
    """
    graph_file = "test_graph1.txt"
    graph = g.Graph(graph_file)
    
    #Test with another manual solution. Swap 3 with 5. The cut size is 5.
    #Start with a bad solution.
    solution = {1: 0, 2: 0, 3: 1, 4: 1, 5: 0, 6: 1}
    graph.set_solution_explicit(solution)
    cut_size = graph.get_cut_size()
    assert cut_size == 5
    
    #If we move the node 3 to the other partition,
    #the cut size will be decreased by 3, so the gain must be 3.
    node3 = graph.nodes[3]  
    gain,cut = graph.calculate_gain(node3)  
    assert gain == 3
    assert node3.last_calculated_gain == gain #The gain is stored in the node object for easy access.
    
    #Move the node to the other partition
    graph.move_node(3)
    assert graph.nodes[3].partition == 0
    cut_size = graph.get_cut_size()
    #cut size is decreased by 3, so cut size will be 2 (because 5 is still there), 
    #the solution is unbalanced at this stage.
    assert cut_size == 2 
    
    #get the gain of 5.
    node5 = graph.nodes[5]
    gain,cut = graph.calculate_gain(node5)
    assert gain == 1 # Gain will be 1 because we moved 3 to the other partition.
    assert node5.last_calculated_gain == gain
    
    #move the node 5 to the other partition
    graph.move_node(5)
    cut_size = graph.get_cut_size()
    #cut size is decreased by 1.
    #so cut size will be 1 and the solution is balanced. This is the optimal solution.
    #This is the optimal convergence behavior for this trivial graph.
    assert cut_size == 1
    
def test_graph_operations():
    """This test case is for the graph operations. 
    It will test the primary node operations like moving a node to the other partition and calculating the gain of a node.
    It is also a mini-simulation of the FM algorithm.
    """
    graph_file = "test_graph1.txt"
    graph = g.Graph(graph_file)
    
    #Test with another manual solution. Swap 3 with 5. The cut size is 5.
    #Start with a bad solution.
    graph.set_solution_explicit({1: 0, 2: 0, 3: 1, 4: 1, 5: 0, 6: 1})
    cut_size = graph.get_cut_size()
    assert cut_size == 5
    assert graph.is_balanced == True
    balance = graph.get_partition_balance()
    assert balance == 0
    
    #If we move the node 3 to the other partition,
    #the cut size will be decreased by 3, so the gain must be 3.
    node3 = graph.nodes[3]  
    gain = graph.calculate_gain(node3)
    assert gain == (3,3) #gain,cut. Note that the cut is 3, because the node 3 is connected to 1,2,5 in the other partition.
    assert node3.last_calculated_gain == gain[0] #The gain is stored in the node object for easy access.
    
    #Move back the gain must be -previous gain, because we made a better solution by moving the node 3.
    graph.move_node(3)
    gain = graph.calculate_gain(node3)
    #Note that the cut is 0, because the node 3 is not connected to any node in the other partition, but the solution is unbalanced.
    assert gain == (-3, 0) 
    assert node3.last_calculated_gain == gain[0] #The gain is stored in the node object for easy access.
    assert graph.is_balanced == False
    balance = graph.get_partition_balance()
    assert balance == 2
    
    cut_size = graph.get_cut_size() 
    assert cut_size == 2
    
    #get the gain of 5.
    node5 = graph.nodes[5]
    gain = graph.calculate_gain(node5)
    assert gain == (1,2) # Gain will be 1 because we moved 3 to the other partition.
    assert node5.last_calculated_gain == gain[0]
    
    #move the node 5 to the other partition. Now the solution is balanced.
    graph.move_node(5)
    cut_size = graph.get_cut_size()
    #cut size is decreased by 1.
    assert cut_size == 1
    assert graph.is_balanced == True
    balance = graph.get_partition_balance()
    assert balance == 0
    
    #Check balance again. Move node 1 to the other partition. Now the solution is unbalanced in fgavor of partition 1.
    graph.move_node(1)
    balance = graph.get_partition_balance()
    assert balance == -2
    assert graph.is_balanced == False

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
    
def test_fm_single_pass():
    #See test_scenario_1.png for the graph.
    #load the test graph
    graph_file = "test_graph1.txt"
    graph = g.Graph(graph_file)
    
    #Setup the scenario. Bad solution.
    custom_solution = {1: 0, 2: 0, 3: 1, 4: 1, 5: 0, 6: 1}
    graph.set_solution_explicit(custom_solution)
    cut_size = graph.get_cut_size()
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
    
    ####### RUN SINGLE PASS. The node 5 will be moved to the right partition. 
    #The solution is not balanced yet. Expected cut size is 2.
    run = fm.run_single_pass()
    #The best gain is at node5 in the left bucket. The gain is 3.
    best_node = graph.nodes[5]
    #The cut size is 5, so the cut size will be 2.
    assert run[0] == 5 # old cut size
    assert run[1] == 2 # new cut size
    assert run[2] == best_node.id
    
    #We need to check the bucket distribution again.
    #CHECK LEFT BUCKET, it is only node 1 and 2, both have gain 0 (unchanged).
    node = fm.get_node_from_left_bucket(0)
    assert node == graph.nodes[1]
    assert node.prev == None
    node = node.next
    assert node == graph.nodes[2]
    assert node.next == None       
    
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
    
    #Ensure the bucket distribution.     
    expected_bucket_left = [1,2]    
    expected_bucket_right = [6,4,3]
    _check_bucket_contents(fm, expected_bucket_left, expected_bucket_right, max)
            
    #Run the pass again. This time the node 3 will be moved to the left partition.
    run = fm.run_single_pass()
    #The best gain is at node3 in the right bucket. The gain is 2.
    best_node = graph.nodes[3]
    #The cut size is 2, so the cut size will be 1.
    assert run[0] == 2 # old cut size
    assert run[1] == 1 # new cut size
    assert run[2] == best_node.id
    
    #We need to check the bucket distribution again.
    #CHECK LEFT BUCKET, it is only node 1 and 2, they both have the gain -2 now, because 3 is moved to this partition.
    node = fm.get_node_from_left_bucket(-2)
    assert node == graph.nodes[1]
    assert node.prev == None
    node = node.next
    assert node == graph.nodes[2]
    assert node.next == None
    
    #Check the right bucket. 4 and 6 now, since the 3 is removed from the bucket.
    node = fm.get_node_from_right_bucket(-1)
    assert node == graph.nodes[6]
    assert node.prev == None
    
    node = node.next
    assert node == graph.nodes[4]
    assert node.next == None
    
    #Ensure the bucket distributions.     
    expected_bucket_left = [1,2]    
    expected_bucket_right = [6,4]
    _check_bucket_contents(fm, expected_bucket_left, expected_bucket_right, max)

def test_fm_run():
    #See test_scenario_1.png for the graph.
    #load the test graph
    graph_file = "test_graph1.txt"
    graph = g.Graph(graph_file)
    
    #Setup the scenario. Bad solution.
    custom_solution = {1: 0, 2: 0, 3: 1, 4: 1, 5: 0, 6: 1}
    graph.set_solution_explicit(custom_solution)
    cut_size = graph.get_cut_size()
    assert cut_size == 5
    
    #Create the FM object, it initializes the buckets.
    fm = FM(graph)
    assert fm.cut_size == 5
    assert fm.size_part_1 == 3
    assert fm.size_part_2 == 3
    assert fm.max_gain == 3  
    
    #Run the FM algorithm. It will run until the cut size is not improved.
    res = fm.run_fm()
    assert res == 1 #The optimal solution is found. The cut size is 1.
    
    #the solution is balanced now.
    assert graph.get_cut_size() == 1
    for i in range(len(fm.gains_left)):
        assert fm.gains_left[i] == None
        assert fm.gains_right[i] == None
    
    #get the first partition from graph
    partition = graph.get_partition(0)
    assert len(partition) == 3
    assert 1 in partition
    assert 2 in partition
    assert 3 in partition
    
    #get the second partition from graph
    partition = graph.get_partition(1)
    assert len(partition) == 3
    assert 4 in partition
    assert 5 in partition
    assert 6 in partition
    
def test_edge_case_start_optimal():
    graph_file = "test_graph1.txt"
    graph = g.Graph(graph_file)
    
    #Setup the scenario. Optimal solution.
    custom_solution = {1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1}
    graph.set_solution_explicit(custom_solution)
    cut_size = graph.get_cut_size()
    assert cut_size == 1
        
    #Create the FM object, it initializes the buckets.
    fm = FM(graph)
    #Run the FM algorithm. It will run until the cut size is not improved.
    res = fm.run_fm()
    assert res == 1 #The optimal solution is found. The cut size is 1.
    
    #the solution is balanced now.
    assert graph.get_cut_size() == 1
    for i in range(len(fm.gains_left)):
        assert fm.gains_left[i] == None
        assert fm.gains_right[i] == None
    
    #get the first partition from graph
    expected_partition = [5,4,6]
    partition = graph.get_partition(0)
    assert expected_partition == partition
    
    #get the second partition from graph
    expected_partition = [1,2,3]
    partition = graph.get_partition(1)
    assert expected_partition == partition
    
def test_graph_with_island():
    #see test_scenario_2.png for the graph.
    graph_file = "test_graph2.txt"
    graph = g.Graph(graph_file)
    #there is an island in the graph. 7,8,9 are connected to each other, separate from the rest of the graph.
    #Setup the scenario. Bad solution.
    custom_solution = {1: 0, 2: 0, 3: 1, 4: 1, 5: 0, 6: 1, 7: 1, 8: 1, 9: 0,10: 0}
    graph.set_solution_explicit(custom_solution)
    cut_size = graph.get_cut_size()
    assert cut_size == 7
    partition1 = graph.get_partition(0)
    assert len(partition1) == 5
    partition2 = graph.get_partition(1)
    assert len(partition2) == 5
    
    fm = FM(graph)
    res = fm.run_fm()
    #The optimal solution is found. The cut size is 3. The island cannot be completely in one partition because of 
    #size constaints.
    assert res == 3
    
    #the solution is balanced now.
    assert graph.get_cut_size() == 3
    for i in range(len(fm.gains_left)):
        assert fm.gains_left[i] == None
        assert fm.gains_right[i] == None
    
    #get the first partition from graph
    expected_partition = [1,2,3,9,10]
    partition1 = graph.get_partition(0)
    assert expected_partition == partition1
    
    #get the second partition from graph
    expected_partition = [5,4,6,7,8]
    partition2 = graph.get_partition(1)
    assert expected_partition == partition2
    
def test_fm_run_500():
    #see test_scenario_2.png for the graph.
    graph_file = "Graph500.txt"
    graph = g.Graph(graph_file)
    solution = {}
    #Initialize a fixed solution. A random solution is not optimal for testing.
    #Iterate over the nodes and assign 0 if the node id is even, 1 if the node id is odd.
    #It is a worse case than random solution or halg-half solution, because it seperates the adjacent nodes,
    #which tend to be in a cluster.
    for node_id in graph.nodes:
        solution[node_id] = node_id % 2
    graph.set_solution_explicit(solution)
    cut_size = graph.get_cut_size()
    assert cut_size == 651
    
    fm = FM(graph)
    res = fm.run_fm()
    assert res == 70
    partition1 = graph.get_partition(0)
    partition2 = graph.get_partition(1)
    assert len(partition1) == 250
    assert len(partition2) == 250
    assert graph.get_cut_size() == 70
    assert graph.is_balanced == True
    stats = fm.get_run_statistics()    
    pass

def _check_bucket_contents(fm:FM, expected_left:list, expected_right:list, max:int):
    #Check left.
    bucket = []
    for gain in range(-max, +max):
        item = fm.get_node_from_left_bucket(gain)
        if item is not None:
            for node in item.to_list():
                bucket.append(node.id)
    assert bucket == expected_left
    
    #Check right.
    bucket = []
    for gain in range(-max, +max):
        item = fm.get_node_from_right_bucket(gain)
        if item is not None:
            for node in item.to_list():
                bucket.append(node.id)
    assert bucket == expected_right  
    
def test_significance():
    """This method tests the implementation of mann-whitney U test. 
    For testing we calculate one sample from recorded data manually and test against api.
    """
    filename="pckl/ils_find_mutation_size/2025-03-26_02-53-50_ILS-mutation_70-runs_10-max_iterations_10000-best_cut_8.8-time_444.173.pkl"
    results_70 = utils.load_pickle(filename)
    #print(results)

    filename="pckl/ils_find_mutation_size/2025-03-26_01-39-48_ILS-mutation_60-runs_10-max_iterations_10000-best_cut_7.2-time_439.55.pkl"
    results_60 = utils.load_pickle(filename)

    columns = ['best_cut_size', 'time_elapsed', 'n_stays_in_local_optimum']
    df_60 = pd.DataFrame(results_60, columns=columns)    
    df_70 = pd.DataFrame(results_70, columns=columns)

    cut_60 = list(df_60['best_cut_size'])[:-1]
    cut_70 = list(df_70['best_cut_size'])[:-1]
    # Create DataFrames for each group
    df1 = pd.DataFrame({'Best Cut': cut_60, 'mutation_size': 60})
    df2 = pd.DataFrame({'Best Cut': cut_70, 'mutation_size': 70})

    # Concatenate vertically and sort
    df_merged = pd.concat([df1, df2])#.sort_values('Best Cut')
    # LLM prompt:add a new column rank to df_merged. give a rank to each Best Cut starting from 1. If the Best Cut value is same, then rank is same. the dataframe is already sorted.
    # Add rank column starting from 1, with ties assigned the same rank
    # NOTE: LLM used min by default, we changed to average. 
    df_merged['rank'] = df_merged['Best Cut'].rank(method='average')
    
    rank_sum_60 = df_merged[df_merged['mutation_size'] == 60]['rank'].sum()
    rank_sum_70 = df_merged[df_merged['mutation_size'] == 70]['rank'].sum()
    
    assert len(cut_60) == len(cut_70)
    n = len(cut_60)
    u_val_60 = rank_sum_60 - n*(n+1)/2
    u_val_70 = rank_sum_70 - n*(n+1)/2
    min_u = min(u_val_60, u_val_70)
    # if alpha = 0.05 and n = 1-, the critical U value is 23
    u_crit = 23
    # For the difference being statistically significant, the min U value must be less than critical value.
    # If less, we reject the null hypothesis (rank means are not different.)
    # In our case the difference is not statistically significant.
    assert min_u > u_crit, "The difference is not to be expected to be statistically significant."
    print(f"The cut size difference between mutation size 60 and 70 is statistically significant.")
    
    # Perform Mann-Whitney U test
    statistic, p_value = utils.perform_mann_whitney_u_test(cut_60,cut_70)
    assert statistic == min_u # make sure that the calculated statistics value is same.
    assert p_value > 0.05, "The difference is not expected to be statistically significant"
    pass
