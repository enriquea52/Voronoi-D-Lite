#!/usr/bin/env python3

import matplotlib.pyplot as plt

class Vertex:
    def __init__(self, node):
        self.id = node
        self.adjacent = {}

        self.x = None # Eucledian position in the 2D plane (x axis)
        self.y = None  # Eucledian position in the 2D plane (y axis)

        self.h = None # Heuristic Value of each vertex
        self.g = float('inf') 

        self.f = [None, None] 
        self.rhs = float('inf')

        

    def __str__(self):
        return str(self.id) + ' adjacent: ' + str([x.id for x in self.adjacent])

    def add_neighbor(self, neighbor, weight=0):
        self.adjacent[neighbor] = weight

    def get_connections(self):
        return self.adjacent.keys()  

    def neighbors(self):
        return [x.id for x in self.adjacent]

    def get_id(self):
        return self.id

    def get_weight(self, neighbor):
        return self.adjacent[neighbor]

class Graph:
    def __init__(self):
        self.vert_dict = {}
        self.num_vertices = 0
        self.limit_axes = None
        self.axes_lims = None
        

    def __iter__(self):
        return iter(self.vert_dict.values())

    def add_vertex(self, node):
        self.num_vertices = self.num_vertices + 1
        new_vertex = Vertex(node)
        self.vert_dict[node] = new_vertex
        return new_vertex

    def get_vertex(self, n):
        if n in self.vert_dict:
            return self.vert_dict[n]
        else:
            return None

    def add_edge(self, frm, to, cost = 0):
        if frm not in self.vert_dict:
            self.add_vertex(frm)
        if to not in self.vert_dict:
            self.add_vertex(to)

        self.vert_dict[frm].add_neighbor(self.vert_dict[to], cost)
        self.vert_dict[to].add_neighbor(self.vert_dict[frm], cost)

    def get_vertices(self):
        return self.vert_dict.keys()

    
    def weight_costs(self):
        weights = []
        for v in self.vert_dict:
            node1 = self.vert_dict[v]
            for node2 in node1.get_connections():
                weights.append(node1.get_weight(node2))
        return weights



    def drawGraph(self,start):
        
        plt.clf()

        start = self.vert_dict[start] 
        plt.scatter(start.x,start.y,linewidths=12,c = 'lightgreen',zorder=3)

        for v1 in self.vert_dict: # Draw edges
            node1 = self.vert_dict[v1]
            for v2 in node1.get_connections():
                node2 = v2
                plt.plot([node1.x,node2.x],[node1.y,node2.y],color='green',zorder=1) # Draw Edge between vertexes
                textx = (node1.x + node2.x) / 2
                texty = (node1.y + node2.y) / 2
                #plt.text(textx, texty , str(node1.get_weight(node2))) # Print Weight



        for v in self.vert_dict: # Draw nodes
            node = self.vert_dict[v]
            plt.scatter(node.x,node.y,linewidths=5,c = 'yellow',zorder=2)
            #plt.text(node.x,node.y, str(node.id),zorder=4) # Print heuristic


        for v in self.vert_dict: # Write info nodes
            node = self.vert_dict[v]

            htextx = node.x - 0.05
            htexty = node.y 
            #plt.text(htextx, htexty, str(node.h),zorder=4) # Print heuristic

            ftextx = htextx + 0.05
            ftexty = htexty - 0.3
            #plt.text(ftextx, ftexty,  "f: " + str(node.f),zorder=4) # Print f key value

            rhstextx = ftextx
            rhstexty = ftexty - 0.3
            #plt.text(rhstextx, rhstexty, "rhs: " + str(node.rhs),zorder=4) # Print rhs value

            gtextx = rhstextx
            gtexty = rhstexty - 0.3
            #plt.text(gtextx, gtexty,  "g: " + str(node.g),zorder=4) # Print g value
            if self.limit_axes:
                plt.axis([0, self.axes_lims, 0, self.axes_lims])
            plt.draw()




