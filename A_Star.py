import math
import pandas as pd
import networkx as nx
import GraphMaker
import matplotlib.pyplot as plt

graph = GraphMaker.MakeGraph()
# nx.draw_networkx(graph)
# plt.show()
# exit()
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

def convert_to_time(x):
    km_h = 45
    return (x / km_h) * 3600
def timeToSecond(current_time):
    list_of_time = current_time.split(':')
    
    time = 0
    for i in range(len(list_of_time)):
            num = int(list_of_time[i])
            time += num * 60 ** (len(list_of_time) - i - 1)
    return time


def h(u, v):
    distance = abs(u[0] - v[0]) + abs(u[1] - v[1])
    return convert_to_time(distance)

def A_Star(graph, source, goal):
    if source not in graph.nodes or goal not in graph.nodes:
        return []

    state = {}
    g = {}
    f = {}
    previous = {}
    minHeap = PriorityQueue()

    for v in graph.nodes:
        state[v] = "unseen"
        g[v] = math.inf
        f[v] = math.inf
        previous[v] = None
    
    g[source] = 25200 
    f[source] = 25200  + h((graph.nodes[source]['lat'],graph.nodes[source]['lon']), (graph.nodes[goal]['lat'],graph.nodes[goal]['lon']))
    state[source] = "open"
    minHeap.insert((source, f[source], g[source]))

    while not minHeap.empty():
        print("Current Heap Before popping", minHeap.data)
        x, _, _ = minHeap.remove_smallest()
        print("Popped Item", x)
        state[x] = "closed"

            
        if x == goal:
            break

        for edge in list(graph.edges(x,data=True)):
            w = edge[1]
            if state[w] != "closed":
                tenative_g = g[x] + edge[2]['weight']
                edge[2]["departure_time"].sort()
                new_departure_time = None
                for dep in edge[2]["departure_time"]:
                    if new_departure_time is None or new_departure_time - g[x] > dep - g[x] and dep - g[x] > 0:
                        new_departure_time = dep
                waitingTime = new_departure_time - g[x]
                if edge[2]["trip_id"][0] == "W":
                    waitingTime = 0
                addDep = new_departure_time if waitingTime > 0 else 0
                
                if   tenative_g + waitingTime < g[w]:
                    j = g[w]
                    g[w] =  new_departure_time if addDep != 0 else tenative_g
                    f[w] = (new_departure_time if addDep != 0 else tenative_g )+ ( h((graph.nodes[w]['lat'],graph.nodes[w]['lon']), (graph.nodes[goal]['lat'],graph.nodes[goal]['lon'])) )
                    print(x,"->",w)
                    print(waitingTime)
                    # print("Before Change", j)
                    # print("After Change", g[w])
                    previous[w] = (x,new_departure_time)
                    state[w] = "open"
                    
                    minHeap.decrease_key((w, f[w], g[w]))
    path = []    
    while goal is not None:
        path.append(goal)
        goal = previous[goal][0] if previous[goal] else None
    return path[::-1]

path = A_Star(graph, "U100Z101P", "U588Z1P")
print("Path:", path)
for edge in list(graph.edges(data=True)):
    print(edge)