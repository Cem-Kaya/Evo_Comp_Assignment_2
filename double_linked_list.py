from collections import defaultdict

class Dll_Node:
    #A node in a double linkedlist.

    def __init__(self, node_id):
        self.node_id = node_id
        self.prev = None
        self.next = None
        
        
        
class Double_Linked_List:
    # push/pop/remove  O(1)

    def __init__(self):
        self.head = None
        self.tail = None

    def is_empty(self):
        return (self.head is None)

    def push_front(self, dll_node):
        dll_node.prev = None
        dll_node.next = self.head
        if self.head is not None:
            self.head.prev = dll_node
        self.head = dll_node
        if self.tail is None:
            self.tail = dll_node

    def pop_front(self):        
        if self.head is None:
            return None
        front_node = self.head
        new_head = front_node.next
        self.head = new_head
        if new_head is not None:
            new_head.prev = None
        else:
            self.tail = None
        front_node.prev = None
        front_node.next = None
        return front_node

    def remove(self, dll_node):        
        prev_node = dll_node.prev
        next_node = dll_node.next

        if prev_node is None:
            # removing head
            self.head = next_node
        else:
            prev_node.next = next_node

        if next_node is None:
            # removing tail
            self.tail = prev_node
        else:
            next_node.prev = prev_node

        dll_node.prev = None
        dll_node.next = None

    def __str__(self):
        # stringfy the double linked list
        string_ver = ""
        cur = self.head
        while cur:
            string_ver += " <-> " + str(cur.node_id )
            cur = cur.next
        return string_ver[5:]     
        
        
if __name__ == "__main__":
    print("Testing Double Linked List")
    dll = Double_Linked_List()
    for i in range(10):
        dll.push_front(Dll_Node(i))
    print(dll)