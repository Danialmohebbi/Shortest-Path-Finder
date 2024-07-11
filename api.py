import json
import pandas as pd
import networkx as nx


def timeToSecond(current_time):
    list_of_time = current_time.split(':')
    print(list_of_time)
    time = 0
    for i in range(len(list_of_time)):
            num = int(list_of_time[i])
            time += num * 60 ** (len(list_of_time) - i - 1)
    return time

j_file = open("stops.json", "r")
j_loaded = json.load(j_file)
# print(j_loaded.keys())
# print(j_loaded["generatedAt"])
# print(len(j_loaded["stopGroups"]))

stops = pd.read_csv("pid_gtfs/stops.txt")
stop_times = pd.read_csv('pid_gtfs/stop_times.txt', low_memory=False)
trips = pd.read_csv('pid_gtfs/trips.txt', low_memory=False)
routes = pd.read_csv('pid_gtfs/routes.txt', low_memory=False)

G = nx.DiGraph()

for index, row in stops.iterrows():
    #print(row['stop_name'], row['stop_lat'])
    G.add_node(row['stop_id'], name=row['stop_name'], lat=row['stop_lat'], lon=row['stop_lat'])
    

for index, row in stop_times.iterrows():
    
    trip_id = row['trip_id']
    stop_id = row['stop_id']
    next_index = index + 1

    if next_index < len(stop_times) and stop_times.iloc[next_index]['trip_id'] == trip_id:
        next_stop_id = stop_times.iloc[next_index]['stop_id']
        departure_time = timeToSecond(stop_times.iloc[index]['departure_time'])
        arrival_time = timeToSecond(stop_times.iloc[next_index]['arraival_time'])
        
        print(arrival_time)
        time = (arrival_time - departure_time) / 60.0 #this includes waiting time
        
        G.add_edge(stop_id, next_stop_id, trip_id=trip_id, departure_time=row['departure_time'], weight=time)
    #print(trip_id)
print(G.edges["U3Z1P"])