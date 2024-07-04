import math

class PriorityQueue:
    def __init__(self):
        self.data = []
        self.indices = {}

    def count(self):
        return len(self.data)

    def decrease_key(self, x):
        if x[0] not in self.indices:
            self.insert(x)
        else:
            index = self.indices[x[0]]
            self.data[index] = x
            self.upHeap(index)
            self.downHeap(index)

    def insert(self, x):
        self.data.append(x)
        index = self.count() - 1
        self.indices[x[0]] = index
        self.upHeap(index)

    def remove_smallest(self):
        if self.count() <= 0:
            raise IndexError

        smallest = self.data[0]
        self.indices.pop(smallest[0])

        if self.count() > 1:
            self.data[0] = self.data.pop()
            self.indices[self.data[0][0]] = 0
            self.downHeap(0)
        else:
            self.data.pop()

        return smallest

    def upHeap(self, i):
        while i > 0:
            p = (i - 1) // 2
            if self.data[p][1] > self.data[i][1]:
                self.data[p], self.data[i] = self.data[i], self.data[p]
                self.indices[self.data[p][0]], self.indices[self.data[i][0]] = p, i
                i = p
            else:
                break

    def downHeap(self, i):
        while True:
            l = 2 * i + 1
            r = 2 * i + 2
            smallest = i
            if l < self.count() and self.data[smallest][1] > self.data[l][1]:
                smallest = l
            if r < self.count() and self.data[smallest][1] > self.data[r][1]:
                smallest = r

            if smallest != i:
                self.data[smallest], self.data[i] = self.data[i], self.data[smallest]
                self.indices[self.data[smallest][0]], self.indices[self.data[i][0]] = smallest, i
                i = smallest
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
    return math.sqrt((u[0] - v[0]) ** 2 + (u[1] - v[1]) ** 2)


def A_Star(graph, source, goal):
    if source not in graph.vertices or goal not in graph.vertices:
        return []

    state = {}
    g = {}
    f = {}
    previous = {}
    minHeap = PriorityQueue()

    for v in graph.vertices:
        state[v] = "unseen"
        g[v] = math.inf
        f[v] = math.inf
        previous[v] = None

    g[source] = 0
    f[source] = h(graph.vertices_cords[source], graph.vertices_cords[goal])
    state[source] = "open"
    minHeap.insert((source, f[source], g[source]))

    while not minHeap.empty():
        print(minHeap.data)
        x, _, _ = minHeap.remove_smallest()
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
                    minHeap.decrease_key((w, f[w], g[w]))
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
