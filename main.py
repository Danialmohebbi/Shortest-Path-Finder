import math


class PriorityQueue:
    def __init__(self):
        self.data = []
        self.indices = {}

    def count(self):
        return len(self.data)

    def decrease_key(self, x):
        if not self.indices.__contains__(x[0]):
            self.insert(x)
        else:
            self.data[self.indices[x[0]]] = x
            self.upHeap(self.indices[x[0]])

    def insert(self, x):
        self.data.append(x)
        self.upHeap(self.count() - 1)

    def remove_smallest(self):
        if self.count() <= 0:
            raise IndexError

        s, self.data[0] = self.data[0], self.data[-1]
        self.data.pop()
        self.downHeap(0)
        return s

    def upHeap(self, i):
        while i > 0:
            p = (i - 1) // 2
            if self.data[p][1] > self.data[i][1]:
                self.data[p], self.data[i] = self.data[i], self.data[p]
                self.indices[self.data[p]] = i
                self.indices[self.data[i]] = p
                i = p
            else:
                break

    def downHeap(self, i):
        while True:
            l = 2 * i + 1
            r = 2 * i + 2
            swap = i
            if l < self.count() and self.data[swap][1] > self.data[l][1]:
                swap = l
            if r < self.count() and self.data[swap][1] > self.data[r][1]:
                swap = r

            if swap != i:
                self.data[swap], self.data[i] = self.data[i], self.data[swap]
                self.indices[self.data[swap]] = i
                self.indices[self.data[i]] = swap
                i = swap
            else:
                break

    def empty(self):
        return len(self.data) == 0


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

    def addVertice(self, v):
        self.vertices.add(v)
        if self.m.get(v) is None:
            self.m[v] = None

    def addEdge(self, source, sink, weight, direction):
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
        return self.m.__contains__(v)

    def has_edge(self, v, s):
        if self.has_vertex(v):
            cur = self.m[v]
            while cur is not None:
                if cur.sink == s:
                    return True
        return False


def Dijkstra(graph, source, goal):
    state = {}
    h = {}
    previous = {}
    minHeap = PriorityQueue()
    for v in graph.vertices:
        state[v] = "unseen"
        h[v] = math.inf
        previous[v] = None

    h[source] = 0
    state[source] = "open"
    minHeap.insert((source, 0))
    while not minHeap.empty():
        print(minHeap.data)
        x, _ = minHeap.remove_smallest()
        state[x] = "closed"
        if x == goal:
            break
        
        edge = graph.m[x]
        while edge is not None:
            w = edge.sink
            if state[w] != "closed":
                if h[w] > h[x] + edge.weight:
                    h[w] = h[x] + edge.weight
                    previous[w] = x
                    state[w] = "open"
                    minHeap.decrease_key((w, h[w]))
            edge = edge.next

    path = []
    while goal is not None:
        path.insert(0, goal)
        goal = previous[goal]

    print(state)
    return path


g = Graph()

g.addEdge("A", "B", 7, True)
g.addEdge("A", "E", 1, True)
g.addEdge("B", "E", 8, True)
g.addEdge("B", "C", 3, True)
g.addEdge("E", "C", 2, True)
g.addEdge("E", "D", 7, True)
g.addEdge("C", "D", 6, True)
g.addEdge("A","G",0,True)
g.addEdge("G","H",0,True)
g.addEdge("H","F",0,True)

g.addVertice("A")
g.addVertice("B")
g.addVertice("C")
g.addVertice("D")
g.addVertice("E")
g.addVertice("G")
g.addVertice("H")
g.addVertice("F")

path = Dijkstra(g, "A", "D")
print("Path:", path)
