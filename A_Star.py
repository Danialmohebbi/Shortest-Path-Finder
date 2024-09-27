import json
import math
import pandas as pd
import networkx as nx
import GraphMaker
import matplotlib.pyplot as plt

graph = GraphMaker.MakeGraph()
#nx.draw_networkx(graph)
#plt.show()
#print(graph.edges["U115Z6P"])
exit()
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
    def __init__(self, source, sink, weight,departure_time, direction, next):
        self.source = source
        self.sink = sink
        self.weight = weight
        self.direction = direction
        self.next = next
        self.departure_time = departure_time


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

    def addEdge(self, source, sink, weight,departure_time, direction):
        if weight < 0:
            raise ValueError("Graph contains negative edge weight, which is not allowed for A* algorithm.")
        if self.m.get(source) is None:
            self.m[source] = Edge(source, sink, weight,departure_time, direction, None)
        else:
            Node = Edge(source, sink, weight,departure_time, direction, self.m[source])
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
                tenative_g = g[x] + edge.weight + edge.departure_time
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

    with open('magic.txt', 'w') as f:
        for key, value in state.items():
            f.write(f"{key}:{value}\n")
    return path

stops = pd.read_csv("pid_gtfs/stops.txt")
stop_times = pd.read_csv('pid_gtfs/stop_times.txt', low_memory=False)
trips = pd.read_csv('pid_gtfs/trips.txt', low_memory=False)
routes = pd.read_csv('pid_gtfs/routes.txt', low_memory=False)
g = Graph()
station = {}
for index, row in stops.iterrows():
    g.addVertice(row['stop_id'],row['stop_lat'], row['stop_lat'])
    station[row['stop_id']] = row['stop_name']

def timeToSecond(current_time):
    list_of_time = current_time.split(':')
    print(list_of_time)
    time = 0
    for i in range(len(list_of_time)):
            num = int(list_of_time[i])
            time += num * 60 ** (len(list_of_time) - i - 1)
    return time

def convert_to_time(x):
    km_h = 50
    return (x / km_h) * 60 

for index, row in stop_times.iterrows():
    
    trip_id = row['trip_id']
    stop_id = row['stop_id']
    
    next_index = index + 1 # type: ignore

    if next_index < len(stop_times) and stop_times.iloc[next_index]['trip_id'] == trip_id:
        next_stop_id = stop_times.iloc[next_index]['stop_id']
        departure_time = timeToSecond(stop_times.iloc[index]['departure_time'])
        arrival_time = timeToSecond(stop_times.iloc[next_index]['arrival_time'])
        
        time = (arrival_time - departure_time) / 60.0
        
        g.addEdge(stop_id, next_stop_id, time, 0, True)

path = A_Star(g, "U536Z3P", "U361Z1P")
for i in range(len(path)):
    path[i] = station[path[i]]  
print("Path:", path)
