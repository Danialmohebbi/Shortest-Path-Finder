from django.shortcuts import render
from django.http import HttpResponse
from datetime import datetime
import networkx as nx
import sys
import os

higher_level_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', ''))
sys.path.append(higher_level_path)

import GraphMaker
def TimeToSecond(current_time):
    list_of_time = current_time.split(':')
    
    time = 0
    for i in range(len(list_of_time)):
            num = int(list_of_time[i])
            time += num * 60 ** (len(list_of_time) - i - 1)
    return time
def index(request):
    now = datetime.now()
    formatted_time = now.strftime("%H:%M:%S")
    time_second = GraphMaker.timeToSecond(formatted_time)
    g = GraphMaker.MakeGraph()
    print(g.edges["U536Z3P","U7406Z3P"])
    return render(request, "life/index.html",{
         "current_time": formatted_time
    })
