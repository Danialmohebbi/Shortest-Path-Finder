import math
from typing import Optional
import math
class Node:
    def __init__(self, key):
        self.key = key[1]
        self.data = key
        self.degree = 0
        self.parent: Optional['Node'] = None
        self.child: Optional['Node'] = None
        self.left: Optional['Node'] = None
        self.right: Optional['Node'] = None
        self.mart = False


class FibHeap:
    def __init__(self):
        self.n = 0
        self.min: Optional['Node'] = None
        self.rootList: Optional['Node'] = None
    
    def minimum(self):
        return self.min.key # type: ignore

    def insert(self, key):
        node = Node(key)
        node.left = node
        node.right = node
        
        self.merge_with_root_list(node)
        if self.min is None or key[1] < self.min.key:
            self.min = node
            
        self.n += 1
        return node
    
    def merge_with_root_list(self,node):
        if self.rootList is None:
            self.rootList = node
        else:
            node.right = self.rootList
            node.left = self.rootList.left
            self.rootList.left.right = node # type: ignore
            self.rootList.left = node
    
    
    def union(self, heap_2):
        new_heap = FibHeap()
        new_heap.n = self.n + heap_2.n
        new_heap.min = self.min if self.min.key < heap_2.min.key else heap_2.min # type: ignore
        new_heap.rootList = self.rootList
        
        first = heap_2.rootList.left
        heap_2.rootList.left = new_heap.rootList.left # type: ignore
        new_heap.rootList.left.right = heap_2.rootList # type: ignore
        new_heap.rootList.left = first # type: ignore
        new_heap.rootList.left.right = new_heap.rootList # type: ignore
        
        return new_heap
    def remove_from_rootList(self, node):
        if node == self.rootList:
            self.rootList = node.right
        node.left.right = node.right
        node.right.left = node.left
    def consolidate(self):
        A = [None] * int(math.log(self.n) * 2)
        nodes = [w for w in self.iterate(self.rootList)]
        for w in range(0, len(nodes)):
            x = nodes[w]
            d = x.degree
            while A[d] != None:
                y = A[d]
                if x.key > y.key:
                    temp = x
                    x, y = y, temp
                self.heap_link(y, x)
                A[d] = None
                d += 1
            A[d] = x
        for i in range(0, len(A)):
            if A[i] is not None:
                if A[i].key < self.min.key: # type: ignore
                    self.min = A[i]
    def merge_with_child_list(self, parent, node):
        if parent.child is None:
            parent.child = node
        else:
            node.right = parent.child.right
            node.left = parent.child
            parent.child.right.left = node
            parent.child.right = node
    def heap_link(self, y, x):
        self.remove_from_rootList(y)
        y.left = y.right = y
        self.merge_with_child_list(x, y)
        x.degree += 1
        y.parent = x
        y.mark = False
    def iterate(self, head):
        node = head
        stop = head
        flag = False

        while True:
            if node == stop and flag is True:
                break
            elif node == stop:
                flag = True
            yield node
            node = node.right
    def extract_min(self):
        z = self.min
        if z is not None:
            if z.child is not None:
                children = [c for c in self.iterate(z.child)]
                for c in children:
                    self.merge_with_root_list(c)
                    c.parent = None
            self.remove_from_rootList(z)
            
            if z == z.right:
                self.min = None
                self.rootList = None
            else:
                self.min = z.right
                self.consolidate()
            self.n -= 1
        
        return z # type: ignore

    def decrease_key(self, x, k):
        if k[1] > x.key:
            return None
        x.key = k[1]
        x.data = k
        y = x.parent
        if y is not None and x.key < y.key:
            self.cut(x, y)
            self.cascading_cut(y)
        if x.key < self.min.key: # type: ignore
            self.min = x
    
    def cut(self, x, y):
        self.remove_from_child_list(y, x)
        y.degree -= 1
        self.merge_with_root_list(x)
        x.parent = None
        x.mark = False

    def cascading_cut(self, y):
        z = y.parent
        if z is not None:
            if y.mark is False:
                y.mark = True
            else:
                self.cut(y, z)
                self.cascading_cut(z)

    def remove_from_child_list(self, parent, node):
        if parent.child == parent.child.right:
            parent.child = None
        elif parent.child == node:
            parent.child = node.right
            node.right.parent = parent
        node.left.right = node.right
        node.right.left = node.left
    
    def empty(self):
        return self.rootList == None

class Edge:
    def __init__(self, source, sink, weight, direction, next):
        self.source = source
        self.sink = sink
        self.weight = weight
        self.direction = direction
        self.next = next


class Graph:
    def __init__(self):
        self.m = {}
        self.vertices = set()
        self.vertices_cords = {}

    def addVertice(self, v, x, y):
        if v not in self.vertices:
            self.vertices.add(v)
            self.vertices_cords[v] = (x, y)
        if self.m.get(v) is None:
            self.m[v] = None

    def addEdge(self, source, sink, weight, direction):
        if weight < 0:
            raise ValueError("Graph contains negative edge weight, which is not allowed for A* algorithm.")
        if self.m.get(source) is None:
            self.m[source] = Edge(source, sink, weight, direction, None)
        else:
            Node = Edge(source, sink, weight, direction, self.m[source])
            self.m[source] = Node

    def getNeighbours(self, source):
        sinks = []
        cur = self.m[source]
        while cur is not None:
            sinks.append(cur.sink)
            cur = cur.next
        return sinks

    def has_vertex(self, v):
        return v in self.m

    def has_edge(self, v, s):
        if self.has_vertex(v):
            cur = self.m[v]
            while cur is not None:
                if cur.sink == s:
                    return True
        return False


def h(u, v):
    return convert_to_time(math.sqrt((u[0] - v[0]) ** 2 + (u[1] - v[1]) ** 2))

def convert_to_time(x):
    km_h = 50
    return int((x / km_h) * 60) #convert to min


def A_Star(graph, source, goal):
    if source not in graph.vertices or goal not in graph.vertices:
        return []

    state = {}
    g = {}
    f = {}
    vertices_nodes = {}
    previous = {}
    minHeap = FibHeap()

    for v in graph.vertices:
        state[v] = "unseen"
        g[v] = math.inf
        f[v] = math.inf
        previous[v] = None

    g[source] = 0
    f[source] = h(graph.vertices_cords[source], graph.vertices_cords[goal])
    state[source] = "open"
    vertices_nodes[source] = minHeap.insert((source, f[source], g[source]))

    while not minHeap.empty():
        node = minHeap.extract_min()
        x = node.data[0] # type: ignore
        state[x] = "closed"

        if x == goal:
            break

        edge = graph.m[x]
        while edge is not None:
            w = edge.sink
            if state[w] != "closed":
                tenative_g = g[x] + edge.weight
                if  tenative_g < g[w]:
                    g[w] = tenative_g
                    f[w] = tenative_g + h(graph.vertices_cords[w], graph.vertices_cords[goal])
                    previous[w] = x
                    state[w] = "open"
                    vertices_nodes[w] = minHeap.insert((w, f[w], g[w]))
                    minHeap.decrease_key(vertices_nodes[w] , (w, f[w], g[w]))
            edge = edge.next

    path = []
    while goal is not None:
        path.insert(0, goal)
        goal = previous[goal]

    print(state)
    return path


g = Graph()
g.addVertice("A", 1, 0)
g.addVertice("B", 2, 0)
g.addVertice("C", 3, 0)
g.addVertice("D", 2, 1)
g.addVertice("E", 4, 0)
g.addVertice("G", 0, 1)
g.addVertice("H", 0, 2)
g.addVertice("F", 0, 0)

g.addEdge("A", "B", 7, True)
g.addEdge("A", "E", 1, True)
g.addEdge("B", "E", 8, True)
g.addEdge("B", "C", 3, True)
g.addEdge("E", "C", 2, True)
g.addEdge("E", "D", 7, True)
g.addEdge("C", "D", 6, True)

path = A_Star(g, "A", "D")
print("Path:", path)
