import json
import math
import pandas as pd
import networkx as nx
import pickle

def timeToSecond(current_time):
    list_of_time = current_time.split(':')
    
    time = 0
    for i in range(len(list_of_time)):
            num = int(list_of_time[i])
            time += num * 60 ** (len(list_of_time) - i - 1)
    return time

# j_file = open("stops.json", "r")
# j_loaded = json.load(j_file)
# print(j_loaded.keys())
# print(j_loaded["generatedAt"])
# print(len(j_loaded["stopGroups"]))

stops = pd.read_csv("pid_gtfs/stops.txt")
stop_times = pd.read_csv('pid_gtfs/stop_times.txt', low_memory=False)
trips = pd.read_csv('pid_gtfs/trips.txt', low_memory=False)
routes = pd.read_csv('pid_gtfs/routes.txt', low_memory=False)

cache_file = 'graph_cache.pkl'
def MakeGraph():
    G = nx.DiGraph()
    try:
            with open(cache_file, 'rb') as f:
                G = pickle.load(f)
            print("Loaded graph from cache.")
    except FileNotFoundError:
            print("No cache file found, creating new graph.")
    #print(G.nodes["U3Z1P"])
    return G
graph = MakeGraph()
# print(graph.nodes["Nádraží Holešovice"])
# print(list(graph.edges("U115Z11P",data=True)))


def convert_to_time(x):
    km_h = 4.8
    return (x / km_h) * 60 
file = "edge_set.txt"
def makeWalkEdges(G, info):
    n = len(info)
    with open(file, "a") as f:
        for v in range(n):
            for w in range(n):
                if v == w:
                    continue
                time = convert_to_time(math.sqrt((info[v][2] - info[w][2]) ** 2 + (info[v][3] - info[w][3]) ** 2))
                G.add_edge(info[v][0],info[w][0],trip_id="W1",departure_time=0,weight=time)
                f.write(f"{info[v][0]},{info[w][0]},trip_id=\"W1\",departure_time=0,weight={time}")
def LoadGraph(G):
    last = None
    info = []
    for index, row in stops.iterrows():
        #print(row['stop_name'], row['stop_lat'])
        G.add_node(row['stop_id'], name=row['stop_name'], lat=row['stop_lat'], lon=row['stop_lat'])
        
        if last is None:
            last = row['stop_name']
            
        if last == row['stop_name']:
            info.append([row['stop_id'], row['stop_name'], row['stop_lat'], row['stop_lat']])
        else:
            makeWalkEdges(G,info)
            info = []
            last = row['stop_name']
            info.append([row['stop_id'], row['stop_name'], row['stop_lat'], row['stop_lat']])
            
        
    count = 0
    for index, row in stop_times.iterrows():
        
        trip_id = row['trip_id']
        stop_id = row['stop_id']
        next_index = index + 1 # type: ignore

        if next_index < len(stop_times) and stop_times.iloc[next_index]['trip_id'] == trip_id:
            next_stop_id = stop_times.iloc[next_index]['stop_id']
            departure_time = timeToSecond(stop_times.iloc[index]['departure_time'])
            arrival_time = timeToSecond(stop_times.iloc[next_index]['arrival_time'])
            
            
            time = (arrival_time - departure_time) / 60.0 #this includes waiting time
            
            G.add_edge(stop_id, next_stop_id, trip_id=trip_id, departure_time=row['departure_time'], weight=time)
            with open(cache_file, 'wb') as f:
                pickle.dump(G, f)
        #print(trip_id)
        count+=1
        print(count)
    