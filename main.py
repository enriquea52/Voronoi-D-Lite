#!/usr/bin/env python3

import site
import d_star_main as dlite
import sys


# Implementation of the D* Algorithm using edges and vertices generated from a Voronoi Diagram
# For robot motion in dynamic environments


# Execution Format
# python main.py start goal sites limit_axex rdm_seed obstacle_sim hid_obs vis_map_nodes

'''
Terminal Execution Arguments:
start: Specifies the starting node where the robot is located. Type of value (int)
goal: Specifies the target node where the robot aims to reach. Type of value (int)
sites: Specifies the number of random point obstacles on the map. Type of value (int)
limit_axes: Specifies if the axes of the map are to be limited, this is specially usefull for visualization purposes. Type of value (0 or 1)
rdm_seed: Specifies a random seed for the ranndom number generator to follow. Type of value (int)
obstacle_sim: It is used to enable or disable the simulation of hidden obstacles in the environment. Type of value (0 or 1)
hid_obs: This is a comma separated list of nodes which end up being hidden obstacles. Type of value (comma separated int values)
vis_map_nodes: This flag is used for only visualization purposes of the node numbers of the resultins voronoi diagram. Type of value (0 or 1)
'''

'''
Execution Examples (Just copy, paste and execute any of the following commands on the terminal):
 - python main.py 1 9 10 0 53 1 4,12
 - python main.py 4 0 20 1 53 1 16,1,18  
 - python main.py 52 75 50 1 53 1 65,4,83,36  
 - python main.py 1 20 20 1 25 1 8,19
 - python main.py 3 19 20 1 30 1 28,22
 - python main.py 1 9 10 0 53 1 4,12,10
 - python main.py 8 87 50 1 25 1 10,21
 - python main.py 8 87 50 1 25 1 10,21,1
 - python main.py 8 87 50 1 25 1 10,21,1,2
 - python main.py 8 87 50 1 25 1 10,21,1,2,84
 - python main.py 10 18 25 1 35 1 23,20,17
'''

if __name__ == '__main__':

    # Retrieve information from the command line ################################################# # 
    if not(len(sys.argv) > 3 or len(sys.argv) < 9):
        print(": ")
        print("Provide at least the following parameters: {start node, goal node, number os point obstacles, limit axes }")
        print("Additional parameters are: specific random seed (int), simulate obstacle, list of hidden obstacles separated by a comma")
        exit()
    elif len(sys.argv) == 5:
        start = int(sys.argv[1])
        goal = int(sys.argv[2])
        sites = int(sys.argv[3])
        limit_axes = bool(int(sys.argv[4]))
        params = [start,goal, sites, limit_axes]
        print("Specified parameters:", params)
        dlite.execute_main(start = start, goal = goal, sites = sites, limit_axes = limit_axes)
    elif len(sys.argv) == 6:
        start = int(sys.argv[1])
        goal = int(sys.argv[2])
        sites = int(sys.argv[3])
        limit_axes = bool(int(sys.argv[4]))
        rdm_seed = int(sys.argv[5])
        params = [start,goal, sites, limit_axes, rdm_seed]
        print("Specified parameters:", params)
        dlite.execute_main(start = start, goal = goal, sites = sites, limit_axes = limit_axes, rdm_seed = rdm_seed)
    elif len(sys.argv) == 8:
        start = int(sys.argv[1])
        goal = int(sys.argv[2])
        sites = int(sys.argv[3])
        limit_axes = bool(int(sys.argv[4]))
        rdm_seed = int(sys.argv[5])
        obs_sim = int(sys.argv[6])
        hid_obs = list(map(int, sys.argv[7].split(',')))
        params = [start,goal, sites, limit_axes, rdm_seed, obs_sim, hid_obs]
        print("Specified parameters:", params)
        dlite.execute_main(start = start, goal = goal, sites = sites, limit_axes = limit_axes, rdm_seed = rdm_seed, obstacle_sim=obs_sim, hid_obs=hid_obs)
    elif len(sys.argv) == 9:
        start = int(sys.argv[1])
        goal = int(sys.argv[2])
        sites = int(sys.argv[3])
        limit_axes = bool(int(sys.argv[4]))
        rdm_seed = int(sys.argv[5])
        obs_sim = int(sys.argv[6])
        hid_obs = list(map(int, sys.argv[7].split(',')))
        vis_map_nodes = int(sys.argv[8])
        params = [start,goal, sites, limit_axes, rdm_seed, obs_sim, hid_obs, vis_map_nodes]
        print("Specified parameters:", params)
        dlite.execute_main(start = start, goal = goal, sites = sites, limit_axes = limit_axes, rdm_seed = rdm_seed, obstacle_sim=obs_sim, hid_obs=hid_obs, vis_map_nodes=vis_map_nodes)
    # ############################################################################################### #
