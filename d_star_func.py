class shortest_path:

    def __init__(self, start, goal, graph):
        self.last = start
        self.km = 0 # key modifier
        self.start = start # Start vertex
        self.goal = goal # Goal vertex
        self.graph = graph # Graph where the algorithm will operate
        self.U = {} # Priority Queue

    def calculateKey(self, s):
        node = self.graph.vert_dict[s]
        node.f[0] = min(node.g, node.rhs) + node.h + self.km
        node.f[1] = min(node.g, node.rhs)
        return node.f

    def initialize(self):
        self.U = {} # Priority Queue
        self.km = 0 # Key modifier
        print("ids")
        for s in self.graph.vert_dict.keys():
            self.graph.vert_dict[s].rhs = float('inf')
            self.graph.vert_dict[s].g = float('inf')
        self.graph.vert_dict[self.goal].rhs = 0 # Set goal's rhs to 0
        self.U[self.goal] = self.calculateKey(self.goal) # Put goal in priority queue with its respective key

    def updateVertex(self, u):
        node = self.graph.vert_dict[u]
        if u != self.goal:
            rhs = min([node.get_weight(s) + s.g for s in node.get_connections()])
            node.rhs = rhs
        if u in self.U:
            del self.U[u]
        if node.g != node.rhs:
            self.U[u] = self.calculateKey(u)
            self.U = dict(sorted(self.U.items(), key=lambda x: x[1][0])) # Sort U list by first key


    def searchPath(self):

        if len(list(self.U)) == 0:
            topkey = [float('inf'),float('inf')]
        else:
            topkey = self.U[list(self.U)[0]]
        kstart = self.calculateKey(self.start)
        snode = self.graph.vert_dict[self.start]

        return ((topkey[0] < kstart[0]) or ((topkey[0] == kstart[0]) and (topkey[1] < kstart[1]))) or (snode.rhs != snode.g)

    def computeShortestPath(self, current_start):
        self.start = current_start

        while self.searchPath():

            u = list(self.U)[0]# U.TopKey implementation
            kold = self.U[u] # U.TopKey implementation

            self.U.pop(u) # U.Pop implementation

            knew = self.calculateKey(u) # current node key

            node = self.graph.vert_dict[u] # Current node

            if (kold[0] < knew[0]) or ((kold[0] == knew[0]) and (kold[1] < knew[1])):
                self.U[u] = self.calculateKey(u)
                self.U = dict(sorted(self.U.items(), key=lambda x: x[1][0])) # Sort U list by first key

            elif node.g > node.rhs:
                node.g = node.rhs
                for s in node.get_connections():
                    self.updateVertex(s.id)

            else:
                node.g = float('inf')
                for s in node.get_connections():
                    self.updateVertex(s.id)
                self.updateVertex(u)
                


            
            






