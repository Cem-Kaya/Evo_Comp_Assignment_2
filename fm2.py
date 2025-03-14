from collections import defaultdict
import random

from graph import Graph
from node import Node
from double_linked_list import * 


class FM:    
    def __init__(self, graph):
        
        assert graph.is_balanced        
        self.graph = graph

        
        self.node_info = {}  # (node_id) -> (dll_ptr, gain, partition, locked)
        self.bucket_left  = defaultdict(Double_Linked_List)
        self.bucket_right = defaultdict(Double_Linked_List)
        self.initialize_buckets_and_gains()
               
    def initialize_buckets_and_gains(self):
        # re init 
        self.node_info.clear()
        self.bucket_left.clear()
        self.bucket_right.clear()
        
        for node_id, node_obj in self.graph.nodes.items():
            gain = self.graph.get_cuts_per_node(node_id)
            part = self.graph.solution[node_id]  # 0 or 1
            locked = False

            new_dll_node = Dll_Node(node_id)
            if part == 0:
                self.bucket_left[gain].push_front(new_dll_node)
            else:
                self.bucket_right[gain].push_front(new_dll_node)

            self.node_info[node_id] = (new_dll_node, gain, part, locked)


    def pick_max_gain_node_respecting_balance(self):
        # if unbalanced pick to make balance 
        # if balanced pick max gain node
        
        best_side = None
        best_gain = None

        if len(self.bucket_left) > len(self.bucket_right):
            best_side = 'left'
        elif len(self.bucket_right) > len(self.bucket_left):
            best_side = 'right'
        else:                   
            max_left_gain = max(self.bucket_left.keys()) 
            max_right_gain = max(self.bucket_right.keys())
            if max_left_gain >= max_right_gain:
                best_side = 'left'
                best_gain = max_left_gain
            elif max_left_gain < max_right_gain :
                best_side = 'right'
                best_gain = max_right_gain                
            else:    
                # random decide tie break      
                best_side = random.choice(['left', 'right'])
                if best_side == 'left':
                    best_gain = max_left_gain
                else:
                    best_gain = max_right_gain

        if best_side is None:
            return (None, 0)  # no buckets at all

        # pop  side  best_gain
        if best_side == 'left':
            dll = self.bucket_left[best_gain]
            # pop front until we find an unlocked node that doesn violate balance
            while not dll.is_empty():
                front_node = dll.pop_front()
                node_id = front_node.node_id
                # check if locked                
                dll_ptr, gain, part, locked = self.node_info[node_id]
                if dll.is_empty():
                    del self.bucket_left[best_gain]

                if locked:
                    # skip locked
                    continue

                if not self.can_move(node_id):
                    # If balance constraints forbid moving, skip it
                    continue

                # found a candidate
                self.node_info[node_id] = (None, gain, part, locked)
                return (node_id, best_gain)

            # if  looped all of the the dll for that gain
            # delete the dictionary key if empty
            if best_gain in self.bucket_left and self.bucket_left[best_gain].is_empty():
                del self.bucket_left[best_gain]

            # no valid node in that bucket, we can keep searching or just return (None,0)
            return (None, 0)

        else:  # best_side == 'right'
            dll = self.bucket_right[best_gain]
            while not dll.is_empty():
                front_node = dll.pop_front()
                node_id = front_node.node_id
                dll_ptr, gain, part, locked = self.node_info[node_id]
                if dll.is_empty():
                    del self.bucket_right[best_gain]

                if locked:
                    continue

                if not self.can_move(node_id):
                    # skip 
                    continue

                self.node_info[node_id] = (None, gain, part, locked)
                return (node_id, best_gain)

            if best_gain in self.bucket_right and self.bucket_right[best_gain].is_empty():
                del self.bucket_right[best_gain]

            return (None, 0)

    def can_move(self, node_id):
        # this is O(N) but can be made O(1) with some bookkeeping !  or can be deleted 
        part = self.graph.solution[node_id]
        
        part0_size = sum(1 for n in self.graph.nodes if self.graph.solution[n] == 0)
        part1_size = sum(1 for n in self.graph.nodes if self.graph.solution[n] == 1)

        if part == 0:
            new_part0_size = part0_size - 1
            new_part1_size = part1_size + 1
        else:
            new_part0_size = part0_size + 1
            new_part1_size = part1_size - 1

  
        diff = abs(new_part0_size - new_part1_size)
        return (diff <= 2)
        
    
    def lock_node(self, node_id):
        # lock node
        dll_ptr, gain, part, _ = self.node_info[node_id]
        self.node_info[node_id] = (dll_ptr, gain, part, True) 
        


    def move_node(self, node_id):
        
        # Flip partition and update neighbor gains in O(1) ! 
        
        old_dll_ptr, old_gain, old_part, locked = self.node_info[node_id]
        new_part = 1 - old_part
        self.graph.solution[node_id] = new_part

        for nbr_id in self.graph.nodes[node_id].neighbors:
            nbr_dll_ptr, nbr_gain, nbr_part, nbr_locked = self.node_info[nbr_id]
            if nbr_dll_ptr is None:
                #print(f"Node {nbr_id} is not in any bucket. Skipping.")
                continue 
            
            if nbr_locked :
                # If nbr is locked 
                continue

            # remove neighbor from old gain bucket in O(1)
            if nbr_part == 0:
                dll = self.bucket_left[nbr_gain]
                dll.remove(nbr_dll_ptr)
                if dll.is_empty():
                    del self.bucket_left[nbr_gain]
            else:
                dll = self.bucket_right[nbr_gain]
                dll.remove(nbr_dll_ptr)
                if dll.is_empty():
                    del self.bucket_right[nbr_gain]

            # Recompute neighbor  gain           
            if self.graph.solution[node_id] == self.graph.solution[nbr_id]:
                new_gain = nbr_gain - 1
            else:
                new_gain = nbr_gain + 1

            # Insert  in the new  bucket
            nbr_dll_ptr.prev = None
            nbr_dll_ptr.next = None
            
            if nbr_part == 0:
                self.bucket_left[new_gain].push_front(nbr_dll_ptr)
            else:
                self.bucket_right[new_gain].push_front(nbr_dll_ptr)

            # Update node_info
            self.node_info[nbr_id] = (nbr_dll_ptr, new_gain, nbr_part, nbr_locked)

        
        self.node_info[node_id] = (None, old_gain, new_part, True)


    def print_buckets(self):
        
        print("Left bucket contents:")
        for g, dll in self.bucket_left.items():
            print(f" Gain={g}: {dll}")
        print("Right bucket contents:")
        for g, dll in self.bucket_right.items():
            print(f" Gain={g}: {dll}")




    def single_pass(self):
        # 1 Re-init
        self.initialize_buckets_and_gains()

        # Data structures for one iter
        move_order = []
        partial_sums = []
        current_sum = 0

        # 2 Keep picking nodes
        while True:
            node_id, gain = self.pick_max_gain_node_respecting_balance()
            if node_id is None:
                # No node can be moved 
                break

            #  move it
            self.move_node(node_id)
            self.lock_node(node_id)
            current_sum += gain
            move_order.append(node_id)
            partial_sums.append(current_sum)

        # 3 Find best partial sum
        if partial_sums:
            best_gain_so_far = max(partial_sums)
            best_index = partial_sums.index(best_gain_so_far)
        else:
            best_index = -1

        # 4 Revert any moves after best_index
        for i in range(best_index+1, len(move_order)):
            # Flip  node back
            self.move_node(move_order[i])
            

        #  return the cut size of the resulting partition
        return self.graph.get_cut_size()
        
    def run_fm(self, max_passes=10):
        # multiple passes
        old_cut_size = self.graph.get_cut_size()
        pass_count = 0

        while pass_count < max_passes:
            pass_count += 1
            print(f"\n--- Starting pass {pass_count} ---")
            new_cut = self.single_pass()
            print(f"Pass {pass_count} produced cut size = {new_cut}")

            if new_cut < old_cut_size:
                print(f"Improved from {old_cut_size} to {new_cut}. Continuing.")
                old_cut_size = new_cut
            else:
                print("No improvement. Stopping.")
                break      
        
        
            
            
if __name__ == "__main__":

    # Load the graph
    graph = Graph("simple_graph.txt")
    graph.set_random_solution(seed=2)
    print(f"graph : {graph}")
    # Construct the FM object
   
    fm = FM(graph)
    fm.print_buckets()
        
    fm.run_fm(max_passes=5)
   
    fm.print_buckets()



    
