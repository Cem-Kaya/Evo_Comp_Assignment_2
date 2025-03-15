from node import Node

#Frst LLM prompt to generate the class skeletion:generate a LinkedNode class. inherit from Node class. It needs next and previous properties.
#We adjusted the rest manually.

class LinkedNode(Node):
    """A node class that inherits from Node and adds linked list functionality"""
    
    def __init__(self, node_id: int, x_in: int = 0, y_in: int = 0):
        super().__init__(node_id, x_in, y_in)
        self.next = None
        self.prev = None
    
    def reset(self):
        """Reset the node's linked list pointers"""
        self.next = None
        self.prev = None
    
    def set_next(self, next_node: 'LinkedNode'):
        """Set the next node in the linked list"""
        self.next = next_node
        if next_node is not None:
            next_node.prev = self
    
    def set_prev(self, prev_node: 'LinkedNode'):
        """Set the previous node in the linked list"""
        self.prev = prev_node
        if prev_node is not None:
            prev_node.next = self
    
    #Remove method: remove the node from the linked list
    def remove(self):
        """Remove the node from the linked list and return the next node
        Returns:
            LinkedNode: The next node in the linked list before removal.
        """
        #We need to return the previous node to keep the iteration going, or next if this is the first note.
        #If this is the only node, return None.        
        ret = self.prev
        if ret is None:
            ret = self.next
            
        if self.prev is not None:
            self.prev.next = self.next
        if self.next is not None:
            self.next.prev = self.prev
            
        self.next = None
        self.prev = None        
        return ret
    
    #insert_after method: insert a new node after the current node
    def insert_after(self, new_node: 'LinkedNode'):
        """Insert a new node after the current node"""
        new_node.prev = self
        new_node.next = self.next
        if self.next is not None:
            self.next.prev = new_node
        self.next = new_node    
    
    def __str__(self):
        # Extend the parent's string representation to include linked list info
        base_str = super().__str__()
        next_id = self.next.id if self.next else None
        prev_id = self.prev.id if self.prev else None
        return f"{base_str[:-1]}, next={next_id}, prev={prev_id})"