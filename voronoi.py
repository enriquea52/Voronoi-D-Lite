#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
from random import randrange
import time
import sys

## Functions for geometry

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])
    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return np.array([x, y])

def medium_perpendicular(p1, p2):
    vector = p2-p1
    medium = (p1+p2)/2
    second_point = medium + [vector[1], -vector[0]]
    return (medium, second_point)

def circle_from_points(p1, p2, p3):
    l1 = medium_perpendicular(p1, p2)
    l2 = medium_perpendicular(p2, p3)
    center = line_intersection(l1, l2)
    return center
    
def contains(a, x):
    return any((a[:]==x).all(1))


# main class
class Volonoi:

    def __init__(self, points):
        points = np.array(points).astype(float)
        self.points = points            # obstacles (sites)
        self.all_points = None          # voronoi vertices + "pointers" for infinite edges 
        self.voronoi_points = None      # indices ov voronoi verices in self.all_points
        self.segments = None            # voronoi segments
        self.processed_point = 0        # last added site index
        self.segments_belongings = None # i-th element is two sites indecies, which border with i-th segments   
        self.init_voronoi()             # initialize voronoi with 3 sites
        self.times = []                 # tracking the time of building the voronoi graph for N sites

    #INITIAL VORONOI DIAGRAM
    def init_voronoi(self):

        p1, p2, p3 = self.points[0:3]
        
        first_vertex = circle_from_points(p1, p2, p3)
        #halflines will belong to these lines
        mp12 = medium_perpendicular(p1, p2)
        mp13 = medium_perpendicular(p1, p3)
        mp23 = medium_perpendicular(p2, p3)
        # find any other point on the line
        d12 = line_intersection(mp12, [p1, p2])
        d13 = line_intersection(mp13, [p1, p3])
        d23 = line_intersection(mp23, [p2, p3])
        # change the point to the oposite if needed
        d12 = self.decide_side(d12, p1, p3, first_vertex)
        d13 = self.decide_side(d13, p1, p2, first_vertex)
        d23 = self.decide_side(d23, p2, p1, first_vertex)
        # add new points
        self.all_points = np.array([first_vertex,
                                    d12,
                                    d13,
                                    d23])
        #only the first one if a voronoi point
        self.voronoi_points = np.array([0])
        self.segments = np.array([[0, 1],
                                  [0, 2],
                                  [0, 3]]).astype(int)
        self.processed_point = 2
        #first middle-prprendicular borders with sites 0 and 1, etc ... 
        self.segments_belongings = [[0, 1], [0, 2], [1, 2]]
    
    def decide_side(self, endpoint, one_point, other_point, center):
        # The closest point of any edges point should be bordering with the edge
        if np.linalg.norm(endpoint - other_point) > np.linalg.norm(one_point - endpoint):
            return center - (center - endpoint)*1000/np.linalg.norm((center - endpoint))
        return center + (center - endpoint)*1000/np.linalg.norm((center - endpoint))
    
    def find_voronoi(self):
        start_time = time.time()
        c = 0

        while self.processed_point < len(self.points)-1:
            #main part, one iteration
            self.process_new_point()
            #time tracking
            new_time = time.time()
            self.times.append(new_time - start_time)
            #just counter output
            c+=1
            if c%20 == 0:
                print(f"{c} interations past...")

    #ADD NEXT POINT TO VORONOI DIAGRAM
    def process_new_point(self):
        ## This is the central function for inserting a new Voronoi site.
        ### The commented plotting lines here are used to create gifs and explainig plot.

        # print("NEW POINTS:")
        # self.plot_voronoi()
        # # self.scatter_new_point()
        # # plt.show()
        # plt.savefig(f"iters/iter{self.processed_point}.png")
        # plt.clf()

        new_pt_indx = self.processed_point + 1  #index of a currently inserted site
        #FIND ALL THE INTERSECTIONS WITH THE SEGMENTS (NEW VORONOI VERTICES)
        segments_intersections, affected_points = self.get_intersections(new_pt_indx)

        # print("NEW POINTS:")
        # self.plot_voronoi()
        # self.scatter_new_point()
        # plt.show()
        
        #CUT INTERSECTED SEGMENTS
        for s_i in segments_intersections.keys():
            new_voronoi_idx = segments_intersections[s_i]
            self.substitude_one_vertex(s_i, new_voronoi_idx, new_pt_indx)
        
        # print("cutted")
        # self.plot_voronoi()
        # self.scatter_new_point()
        # plt.show()
        
        #ADD NEW SEGMENTS
        for p_i in affected_points.keys():
            intersected_segments = affected_points[p_i]
            self.add_new_segment(p_i, segments_intersections, intersected_segments)

        # print("with new segments")
        # self.plot_voronoi()
        # self.scatter_new_point()
        # plt.show()

        #DELETE OLD SEGMENTS THAT ARE IN THE NEW REGION
        for p_i in affected_points.keys():
            self.delete_old_segmets(p_i)
        
        # print("without old segments")
        # self.plot_voronoi()
        # self.scatter_new_point()
        # plt.show()

        self.processed_point += 1


    def add_new_segment(self, p_i, segments_intersections, intersected_segments):
        new_pt_indx = self.processed_point + 1

        #non-infinite segment, just append a segment with ends in the intersection points
        if len(intersected_segments) == 2:
            s1, s2 = intersected_segments
            pv1_i = segments_intersections[s1]
            pv2_i = segments_intersections[s2]
            self.append_segment(pv1_i, pv2_i)
            self.segments_belongings.append([p_i, new_pt_indx])
        
        #infinite segment (a beam), append a segment with one end in the intersection point, and no another end
        if len(intersected_segments) == 1:

            s = intersected_segments[0]
            pv_i = segments_intersections[s]
            # find a line along which will bw the segmwnt and any other point on the line
            mp = medium_perpendicular(self.points[p_i], self.points[new_pt_indx])
            endpoint = line_intersection(mp, [self.points[p_i], self.points[new_pt_indx]])

            # find pint another then the checked one
            bp1, bp2 = self.segments_belongings[s]
            if bp1 == p_i:
                other_point_i = bp2
            else:
                other_point_i = bp1
            other_point = self.points[other_point_i]
            one_point = self.points[p_i]
            # switch the diresction of the halfline if needed
            endpoint = self.decide_side(endpoint, one_point, other_point, self.all_points[pv_i])
            #append an infinite segment
            inf_pt_indx = self.append_infinite_point(endpoint)
            self.append_segment(pv_i, inf_pt_indx)
            self.segments_belongings.append([p_i, new_pt_indx])

    def delete_old_segmets(self, p_i):
        p = self.points[p_i]

        new_pt_indx = self.processed_point + 1
        new_p = self.points[new_pt_indx]
        # gat all the segments of a given point 
        segments = self.get_segments_of_point(p_i)

        for s_i in segments:
            # get segment's ends
            p1_i, p2_i = self.segments[s_i]
            p1, p2 = self.all_points[p1_i], self.all_points[p2_i]
            # If any of the points is closer to the new point, it is inside of the new region
            d1_p = np.linalg.norm(p1 - p)
            d1_np = np.linalg.norm(p1 - new_p)

            d2_p = np.linalg.norm(p2 - p)
            d2_np = np.linalg.norm(p2 - new_p)

            if (d1_np - d1_p) < -0.001 and (d2_np - d2_p)< -0.001:
                self.segments_belongings[s_i] = []

    def get_intersections(self, new_pt_indx):
        # get all new Voronoi points as a result of insertion a new obstacle
        # plt.scatter(new_pt[0], new_pt[1], color = 'red')
        segments_intersections = dict()
        affected_points = dict()
        # check every segment and check if it intersects
        for s_i in range(len(self.segments)):
            ints, int_point = self.process_segment(s_i, new_pt_indx)
            # if intersects, add all the needed inforamtion
            if ints:
                # plt.scatter(int_point[0], int_point[1], color="black")
                new_vor_indx = self.handle_segment_intersection(s_i, int_point, affected_points)
                segments_intersections[s_i] = new_vor_indx
        return segments_intersections, affected_points

    def process_segment(self, segment_index, point_index):
        ## check if the segment is intersected by the newly inserted point's segments
        if len(self.segments_belongings[segment_index]) != 2: # if the segment is already deleted
            return False, None

        # Get middle perpendicular between the new segment and any point bordering with the segment
        other_point_index = self.segments_belongings[segment_index][0]
        other_point = self.points[other_point_index]
        new_point = self.points[point_index]
        mid_perp = medium_perpendicular(new_point, other_point)
        # check if they intersect
        inters, int_point = self.intersect_seg_perp(self.segments[segment_index], mid_perp)
        return inters, int_point
    
    
    def substitude_one_vertex(self, seg_i, vor_pt_index, new_pt_index):
        ## Substitute on vertex of the intersected segment

        old_pt_index = self.segments_belongings[seg_i][0]
        p_inds = self.segments[seg_i]
        # get any real vertex (not the halfline's pointer)
        if p_inds[0] in self.voronoi_points:
            i_non_inf = 0
        else: 
            i_non_inf = 1
        vor_p = self.all_points[p_inds[i_non_inf]]

        new_point = self.points[new_pt_index] # new insernted site (obstacle) (point)
        old_point = self.points[old_pt_index] # any of two old obstacles, previously that border with the intersegment segment

        # point that is in the new region should be substituted 
        d1 = np.linalg.norm(vor_p - new_point)
        d2 = np.linalg.norm(vor_p - old_point)
        if d1 < d2:
            self.segments[seg_i][i_non_inf] = vor_pt_index
        else:
            self.segments[seg_i][1-i_non_inf] = vor_pt_index

    def handle_segment_intersection(self, s_i, int_point, affected_points):
        new_vor_pt_index = self.append_voronoi_point(int_point)
        affectd_cur_points = self.segments_belongings[s_i]

        if affectd_cur_points[0] in affected_points.keys():
            # if the key is already in dictionary
            affected_points[affectd_cur_points[0]].append(s_i)
        else:
            # if the key is not in the dictionary
            affected_points[affectd_cur_points[0]] = [s_i]

        # Same but for the second point
        if affectd_cur_points[1] in affected_points.keys():
            affected_points[affectd_cur_points[1]].append(s_i)
        else:
            affected_points[affectd_cur_points[1]] = [s_i]

        return new_vor_pt_index
        
    def append_voronoi_point(self, point):
        #append an ordinary segment
        self.all_points = np.append(self.all_points, [np.array(point)], axis = 0)
        index = len(self.all_points) - 1
        self.voronoi_points = np.append(self.voronoi_points, index)
        return index
    
    def append_infinite_point(self, point):
        # append a point that shows the direction of the halfline
        self.all_points = np.append(self.all_points, [np.array(point)], axis = 0)
        index = len(self.all_points) - 1
        return index
    
    def append_segment(self, p1_i, p2_i):
        self.segments = np.append(self.segments, [[p1_i, p2_i]], axis = 0)
        index = len(self.segments) - 1
        return index

    def get_segments_of_point(self, p_i):
        n_seg = len(self.segments_belongings)
        return [j for j in  range(n_seg) if p_i in self.segments_belongings[j]]
                    

    def intersect_seg_perp(self, seg, perp):
        # intersect the segment with the middle perpendicular
        p1 = self.all_points[seg[0]]
        p2 = self.all_points[seg[1]]
        is_halfline = not ( (seg[1] in self.voronoi_points) and (seg[0] in self.voronoi_points) )
        if not (seg[0] in self.voronoi_points):
            p1, p2 = p2, p1
        int_point = line_intersection([p1, p2], perp)
        if self.is_point_between(int_point, p1, p2, is_halfline):
            return True, int_point
        return False, None

    def is_point_between(self, pt, left_point, right_point, is_halfline = False):
        # check if the point belongs to the segment
        v1 = pt - left_point
        v2 = right_point - pt
        if is_halfline == True:
            v2 = right_point - left_point
        return sum(v1*v2) > 0

    def is_halfline(self, s):
        # check if the segment is infinite or not
        if not (s[0] in self.voronoi_points and s[1] in self.voronoi_points):
            return True
        return False

    def get_points_and_segments(self, include_infinite = False):
        # get finite virtices and edges of the graph of the FOUND Voronoi
        valid_segments = []
        for s_i in range(len(self.segments)):
            if len(self.segments_belongings[s_i]) == 2:
                if include_infinite or not self.is_halfline(self.segments[s_i]):
                    valid_segments.append(self.segments[s_i])
        voronoi_verticies_i = set()
        for s in valid_segments:
            voronoi_verticies_i.add(s[0])
            voronoi_verticies_i.add(s[1])
        voronoi_verticies_i = list(voronoi_verticies_i)
        voronoi_verticies = [self.all_points[i] for i in voronoi_verticies_i]
        valid_segments = [ [voronoi_verticies_i.index(s[0]), voronoi_verticies_i.index(s[1])] for s in valid_segments]
        return voronoi_verticies, valid_segments


    ###  PLOTTING FUNCTIONS

    def plot_voronoi(self):
        processed_points = self.points[:self.processed_point + 1]
        plt.scatter(processed_points[:, 0], processed_points[:, 1], color = "red", s=7)
        unprocessed_points = self.points[self.processed_point + 1:]
        plt.scatter(unprocessed_points[:, 0], unprocessed_points[:, 1], color = "blue", alpha=0.3, s=7)

        for s_i in range(len(self.segments)):
            if len(self.segments_belongings[s_i]) == 2:
                self.plot_segment(self.segments[s_i])
        
        plt.xlim((-2, 12))
        plt.ylim((-2, 11))
        plt.gca().set_aspect('equal', adjustable='box')
        plt.draw()
        

    def plot_segment(self, seg):
        color = "green"
        p1 = self.all_points[seg[0]]
        p2 = self.all_points[seg[1]]
        if self.is_halfline(seg):
            color = "blue"

        plt.xlim((0, 10))
        plt.ylim((0, 10))
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color = color)

    
    
    def scatter_new_point(self):
        new_pt = self.points[self.processed_point + 1]
        plt.scatter(new_pt[0], new_pt[1], color = 'red', s=7)

## helping function
def generate_points(n_points, seed = None):
    sd = seed
    if seed is None:
        sd = randrange(100)
        print("seed is ", sd)
    np.random.seed(sd)
    points = np.random.random((n_points, 2))*10
    return points

# reading script's parameter function
def get_int_parameter(x):
    try:
        return int(x)
    except:
        print(f"The value {x} must be integer")
        exit(1)

def read_nPoints_parameter_voronoi_script():
    n_points = 50
    if len(sys.argv) == 2:
        n_points = get_int_parameter(sys.argv[1])
    if len(sys.argv) > 2:
        print("Too many parameters. Usage: %s [ n_points ]" % sys.argv[0])
        exit(1)
    return n_points

# program body
if __name__ == '__main__':
    n_points = read_nPoints_parameter_voronoi_script()
    points = generate_points(50, seed=53)

    v = Volonoi(points)
    v.find_voronoi() # find Voronoi Graph

    v.plot_voronoi() # plot the Voronoi Graph
    plt.show()

    verts, segs = v.get_points_and_segments()

    print("Vertices:\n", verts)
    print("Edges:\n", segs)

    plt.plot(v.times)
    plt.show()
