#!/usr/bin/env python3

import numpy
import matplotlib.pyplot as plt
import time
import numpy as np


import graph_theory
import d_star_func
from voronoi import Volonoi, generate_points


def drawShortestPath(g,start,goal):
    path = [start]

    while(path[-1] != goal):
        node = g.vert_dict[path[-1]]
        connections = [ (s.id , node.get_weight(s) + s.g) for s in node.get_connections()]
        next = min(connections, key=lambda x: x[1])[0]
        path.append(next)
    for i in range(0,len(path)-1):
        plt.plot([g.vert_dict[path[i]].x,g.vert_dict[path[i+1]].x], [g.vert_dict[path[i]].y,g.vert_dict[path[i+1]].y],color='blue',zorder=5)

def move_robot(g,start):
    node = g.vert_dict[start]
    connections = [ (s.id , node.get_weight(s) + s.g) for s in node.get_connections()]
    new_start = min(connections, key=lambda x: x[1])[0]
    get_heuristics(g,new_start)
    return new_start

def get_heuristics(g,start): # For this example set the heuristic of every node according to where the robot is located
    s_node = g.vert_dict[start]
    for v in g:
        v.h = eucledian(v.x,v.y,s_node.x,s_node.y)

def eucledian(x1,y1,x2,y2):
    return (((x1-x2)**2) + ((y1-y2)**2))**(0.5)

def create_obstacle(g,s):
    node1 = g.vert_dict[s]
    for node2 in node1.get_connections():
        g.add_edge(node1.id, node2.id, float('inf'))
    
def obstacle_simulation(g,current_node, hidden_obs):

    node = g.vert_dict[current_node]
    connections = [ (s.id , node.get_weight(s) + s.g) for s in node.get_connections()]
    next = min(connections, key=lambda x: x[1])[0]
    if next in hidden_obs:
        create_obstacle(g,next)
        return next

def draw_obstacles(obstacles):
    plt.scatter(obstacles[:,0],obstacles[:,1],linewidths=2, zorder=4)

def init_edges(g,ver,edg):

    for edge in edg:
        g.add_edge(edge[0], edge[1], eucledian(ver[edge[0]][0],ver[edge[0]][1],ver[edge[1]][0],ver[edge[1]][1]))  

def init_vertexes(g,ver):

    for i in range(0, len(ver)):
        g.add_vertex(i)
        g.vert_dict[i].x = ver[i][0]
        g.vert_dict[i].y = ver[i][1]

def map_visualizer(vertexes, edges, limit_axes, axes_lims):
    for s_i in range(len(edges)):
        seg = edges[s_i]
        i, j = seg
        p1, p2 = vertexes[i], vertexes[j]
        plt.text(p1[0], p1[1], i)
        plt.text(p2[0], p2[1], j)
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]])

    if limit_axes:
            plt.axis([0, axes_lims, 0, axes_lims])

    plt.show()
    exit()

def execute_main(start = 1, goal= 9, sites = 10 ,rdm_seed = 53, obstacle_sim = 1, hid_obs = [], limit_axes = False, axes_lims = 10, vis_map_nodes = False):

    g = graph_theory.Graph()
    g.limit_axes = limit_axes
    g.axes_lims = axes_lims

    point_obs = generate_points(sites, seed = rdm_seed)
    voronoi = Volonoi(point_obs)
    voronoi.find_voronoi()
    vertexes, edges = voronoi.get_points_and_segments()

    
    if vis_map_nodes:
        map_visualizer(vertexes, edges, limit_axes, axes_lims)

    init_vertexes(g,vertexes)     # Set vertexes
    init_edges(g,vertexes, edges) # Set Initial edges values


    initial_weights = g.weight_costs() # Get the initial edges costs


    master_piece = d_star_func.shortest_path(start,goal, g)

    get_heuristics(g,start) # Set Initial heuristics

    master_piece.initialize()
    master_piece.computeShortestPath(start)
    g.drawGraph(start)                  # Visualize the initial path
    draw_obstacles(point_obs)
    

    #################################################################################################
    #### Main Loop ##################################################################################
    #################################################################################################

    while start != goal:

 
        
        
        if g.vert_dict[start].g == float('inf'): # If there is no feasible path from the start node to the goal, end the program (there is no solution)
            print("No feasible path found")
            exit()

        drawShortestPath(g,start,goal)
        start = move_robot(g,start) # Move robot to next location, it is important to update the heuristic accordingly

        print("Robot moved to node : ",start)
        

        # Obstacle simulation
        if(obstacle_sim):
            obstacle = obstacle_simulation(g,start,hid_obs)
            if obstacle != None:
                point_obs = np.append(point_obs,[vertexes[obstacle]], axis = 0)
                
        
        new_weights = g.weight_costs() # Scan for graph changes


        if (initial_weights != new_weights):
            master_piece.km = master_piece.km + eucledian(g.vert_dict[master_piece.last].x,g.vert_dict[master_piece.last].y,g.vert_dict[start].x,g.vert_dict[start].y)
            master_piece.last = start
            print("Map updated")

            master_piece.updateVertex(obstacle)
            [master_piece.updateVertex(u.id) for u in g.vert_dict[obstacle].get_connections()]
            
            plt.waitforbuttonpress()
            g.drawGraph(start)
            draw_obstacles(point_obs)

            master_piece.computeShortestPath(start)

        else:
            print("No obstacle detected")

        initial_weights = new_weights 

        plt.waitforbuttonpress()
        
        g.drawGraph(start)
        draw_obstacles(point_obs)
        
    
    print("Goal reached!!")

    plt.show()