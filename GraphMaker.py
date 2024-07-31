import json
import pandas as pd
import networkx as nx
import pickle
import math

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

stops = pd.read_csv("test_gtfs/stops.txt")
stop_times = pd.read_csv('test_gtfs/stops_times.txt', low_memory=False)
# trips = pd.read_csv('pid_gtfs/trips.txt', low_memory=False)
# routes = pd.read_csv('pid_gtfs/routes.txt', low_memory=False)

cache_file = 'test_graph_cache.pkl'
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
#graph = MakeGraph()
def convert_to_time(x):
    km_h = 4.8
    return (x / km_h) * 3600 
file = "edge_set.txt"
def makeWalkEdges(G, info):
    n = len(info)
    with open(file, "a") as f:
        count = 0
        for v in range(n):
            for w in range(n):
                if v == w:
                    continue
                trip_num  = "W" + str(count)
                time = convert_to_time(math.sqrt((info[v][2] - info[w][2]) ** 2 + (info[v][3] - info[w][3]) ** 2))
                G.add_edge(info[v][0],info[w][0],trip_id=trip_num,departure_time=[0],weight=time)
                f.write(f"{info[v][0]},{info[w][0]},trip_id=\"{trip_num}\",departure_time=0,weight={time}\n")
                count+=1
def LoadGraph(G):
    last = None
    info = []
    nodes = set()
    for index, row in stops.iterrows():
        #print(row['stop_name'], row['stop_lat'])
        G.add_node(row['stop_id'], name=row['stop_name'], lat=row['stop_lat'], lon=row['stop_lat'])
        
        if last is None:
            last = row['stop_name']
            nodes.add(row['stop_name'])
            
        if last == row['stop_name']:
            info.append([row['stop_id'], row['stop_name'], row['stop_lat'], row['stop_lat']])
        elif last != row['stop_name'] and row['stop_name'] not in nodes:
            nodes.add(row['stop_name'])
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
            if G.has_edge(stop_id, next_stop_id) and G.edges[stop_id, next_stop_id]['trip_id'][0] != 'W':
                G.edges[stop_id, next_stop_id]['departure_time'].append(departure_time)
            else:
                
                arrival_time = timeToSecond(stop_times.iloc[next_index]['arrival_time'])
                time = (arrival_time - departure_time) 
                G.add_edge(stop_id, next_stop_id, trip_id=trip_id, departure_time=[departure_time], weight=time)
            with open(cache_file, 'wb') as f:
                pickle.dump(G, f)
        #print(trip_id)
        count+=1
        print(count)
    #print(G.edges["U3Z1P"])
# LoadGraph(MakeGraph())
