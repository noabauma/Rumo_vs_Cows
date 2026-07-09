import sys
import time
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi

"""
This code computes the problem of Rumo having to pass a field with n cows.
We have to get from point A to point B with as little contact with the cows as possible, as he starts barking otherwise.
I.e. we are searching for the path of least resistance.
"""

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1] * n  # track tree sizes

    def find(self, i):
        if self.parent[i] != i:
            # Path compression: make parent point directly to root
            self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        ir, jr = self.find(i), self.find(j)
        if ir == jr:
            return False

        # Union by size: attach the smaller tree to the bigger one
        if self.size[ir] < self.size[jr]:
            self.parent[ir] = jr
            self.size[jr] += self.size[ir]
        else:
            self.parent[jr] = ir
            self.size[ir] += self.size[jr]

        return True

def cost_function(a: np.ndarray, b: np.ndarray, c1: np.ndarray, c2: np.ndarray):
    """Cost function to determine the clearance of this edge,
    i.e. the distance from the edge to the nearest cow (the higher the safer).

    Args:
        a (np.ndarray): Starting coordinates in 2d
        b (np.ndarray): End coordinate in 2d
        c1 (np.ndarray): cow 1 coordinate in 2d
        c2 (np.ndarray): cow 2 coordinate in 2d

    Returns:
        cost (float): The clearance of this edge
    """
    m = (c1 + c2)/2 # middle point between the two cows

    # Every point on a voronoi ridge is equidistant from its two generating cows,
    # so all distances can be measured from c1 alone (c2 would give the same).
    # The projection of c1 onto the line through a and b is exactly m, hence we only
    # have to check if m actually lies on the edge. If not, the endpoint (a or b)
    # closest to the cow determines the clearance of the edge.

    # If t in [0, 1]: m is in between, if t in (-inf, 0): m closer to a and if t in (1, inf): closer to b

    t = np.dot(b - a, m - a)/np.dot(b - a, b - a)
    
    if 0 <= t <= 1:
        cost = np.linalg.norm(c1 - m)
    elif t < 0:
        cost = np.linalg.norm(c1 - a)
    else:
        cost = np.linalg.norm(c1 - b)
    
    return cost
    
def compute_graph(vor: Voronoi, obst_coord: np.ndarray, n_obst: int, x_length: float, y_length: float, start_coord: float, end_coord: float):
    """Computing the weighted graph from the voronoi diagram

    Args:
        vor (Voronoi): The voronoi diagram of the (mirrored) cow field
        obst_coord (np.ndarray): The 2d coordinates of all the cows (incl. the mirrored ones)
        n_obst (int): The number of original (unmirrored) cows
        x_length (float): The x length of the field [m]
        y_length (float): The y length of the field [m]
        start_coord (float): The x coordinate of the start point on the lower boundary
        end_coord (float): The x coordinate of the end point on the upper boundary

    Returns:
        np.ndarray: the edges with their weights, sorted by clearance (safest first)
        np.ndarray: mapping from graph node indices to voronoi vertex indices
                    (the start node comes first, the end node last)
    """

    # We start with computing all the middle points of the ridges (also add weight and other important stuff)
    # one middle_point consists of: [start_idx, end_idx, cost, x_coord, y_coord]
    # start_idx: starting ridge_point idx
    # end_idx: ending ridge_point idx
    # cost: the clearance of this edge
    # x_coord: of the middle point between the two cows (debugging purpose)
    # y_coord: of the middle point between the two cows (debugging purpose)
    edges_w_weights = []
    for i, ridge_point in enumerate(vor.ridge_points):
        # go through all the ridge_points with at least one being inside the boundary
        if not (ridge_point[0] >= n_obst and ridge_point[1] >= n_obst):
            assert vor.ridge_vertices[i][0] != -1, "Somehow, this vertex has only one voronoi node. Should not happen with Jonah's mirroring technique!"
            
            middle_point = np.empty((5))

            middle_point[0:2] = vor.ridge_vertices[i]
            
            middle_point[2] = cost_function(vor.vertices[vor.ridge_vertices[i][0]], vor.vertices[vor.ridge_vertices[i][1]], obst_coord[ridge_point[0]], obst_coord[ridge_point[1]])
            middle_point[3:] = np.array(obst_coord[ridge_point[0]] + obst_coord[ridge_point[1]])/2
            
            edges_w_weights.append(middle_point)
            
            
    edges_w_weights = np.array(edges_w_weights)

    # Sort the edges by their clearance, safest (highest) first
    edges_w_weights = edges_w_weights[np.argsort(edges_w_weights[:, 2])[::-1]]

    # First, make a mapping of the vor.vertices to arange as the graph nodes start from 0, n_points -1.
    all_idx = np.unique(edges_w_weights[:,0:2]).astype(int)

    # We define the end and starting point by swapping entries of all_idx! (amazing)
    # The first index is the starting point and the last index the end point
    # The starting point will start on a point on the lower boundary
    # and the end point on a point on the upper boundary
    # We choose the start/endpoints which are the closest to the ridge point on the respective boundaries
    closest_to_start = (-1, np.inf)
    closest_to_end = (-1, np.inf)
    for idx in all_idx:
        vor_vertex = vor.vertices[idx]
        if abs(vor_vertex[1]) < 1e-6:
            dist = abs(vor_vertex[0] - start_coord)
            
            if dist < closest_to_start[1]:
                closest_to_start = (idx, dist)
                
        elif abs(vor_vertex[1] - y_length) < 1e-6:
            dist = abs(vor_vertex[0] - end_coord)
            
            if dist < closest_to_end[1]:
                closest_to_end = (idx, dist)
                
    assert closest_to_start[0] != -1, "didn't find a closest point on the lower boundary"
    assert closest_to_end[0] != -1, "didn't find a closest point on the upper boundary"   
    
    start_idx = np.where(all_idx == closest_to_start[0])[0][0]
    end_idx = np.where(all_idx == closest_to_end[0])[0][0]

    all_idx[0], all_idx[start_idx] = all_idx[start_idx], all_idx[0]

    # the first swap may have moved the end vertex away from position 0
    if end_idx == 0:
        end_idx = start_idx

    all_idx[-1], all_idx[end_idx] = all_idx[end_idx], all_idx[-1]

    return edges_w_weights, all_idx


def union_find(edges_w_weights: np.ndarray, all_idx: np.ndarray):
    """Union-Find: insert the edges, safest first, until the start and end node are connected

    Args:
        edges_w_weights (np.ndarray): the edges [node_i, node_j, cost, ...] sorted by clearance, safest first
        all_idx (np.ndarray): the actual node indices of the voronoi mesh

    Returns:
        defaultdict: adjacency list of all the inserted edges
        int: the number of inserted edges
    """
    # We create a mapping between indices {0, n-1} and the real all_idx for O(1) lookup time
    idx_map = {val: idx for idx, val in enumerate(all_idx)}

    n_nodes = len(all_idx)
    uf = UnionFind(n_nodes)
    graph = defaultdict(list)

    n_edges = edges_w_weights.shape[0]
    edge_idx = 0
    while uf.find(0) != uf.find(n_nodes - 1):
        assert edge_idx < n_edges, f"We did not find a path! (Impossible?!) {edge_idx} == {n_edges}"

        i = idx_map[int(edges_w_weights[edge_idx, 0])]
        j = idx_map[int(edges_w_weights[edge_idx, 1])]

        uf.union(i, j)

        graph[i].append(j)
        graph[j].append(i)

        edge_idx += 1

    return graph, edge_idx

def find_path(graph: defaultdict, n_nodes: int):
    """Find a path from the start node 0 to the end node n_nodes - 1 with a DFS.
    It does not matter which path we find: every inserted edge keeps at least the
    bottleneck clearance, so any path is maximally far away from the cows.

    Args:
        graph (defaultdict): adjacency list of all the inserted edges
        n_nodes (int): the number of nodes of the voronoi mesh

    Returns:
        list: the path from the start node to the end node
        list: all visited nodes, in visiting order (the order is used by the manim scenes)
    """
    visited = []            # keeps the visiting order for the manim scenes
    visited_set = set()     # for O(1) membership tests
    stack = [(0, [0])]      # (current_node, path_so_far)

    while stack:
        node, path = stack.pop()
        if node in visited_set:
            continue
        visited.append(node)
        visited_set.add(node)

        if node == n_nodes - 1:
            return path, visited  # found target

        # push the neighbours in reverse so they are explored in their original order
        for neighbor in reversed(graph[node]):
            if neighbor not in visited_set:
                stack.append((neighbor, path + [neighbor]))

    raise AssertionError("Did not find a path!")
    
    
def main():
    """
    The main function to run the whole algorithm.
    It consists of:

    1. Defining the problem field (#cows, dimensions, start/end point, ...)
    2. Computing the Voronoi diagram
    3. Computing the clearance of the edges and sorting them, safest first
    4. Union-Find a connection from start to end
    5. Finding a path (any path through the inserted edges keeps the maximal clearance)
    6. Plot
    """
    
    time_start = time.time()
    
    ##### Step 1: Defining the problem field
    if len(sys.argv) > 1:
        n_obst = int(sys.argv[1])
        seed = int(sys.argv[2])
        x_length = int((n_obst*100)**0.5)
        y_length = x_length
         
        np.random.seed(seed)
    else:
        x_length = 50        # x length of the cows field [m]
        y_length = 100        # y length of the cows field [m]
        n_obst = 100          # number of obstacles (cows)

        np.random.seed(43)   # seed for the random number generator

    obst_coord = np.random.rand(n_obst, 2) # 2d coordinates of the cows
    obst_coord[:,0] *= x_length
    obst_coord[:,1] *= y_length

    # Mirroring the cow field as we also need the voronoi edges on the boundaries
    # top
    top = np.array((obst_coord[:,0],2*y_length-obst_coord[:,1])).T
    
    # left
    left = np.array((-obst_coord[:,0],obst_coord[:,1])).T
    
    # right
    right = np.array((2*x_length-obst_coord[:,0],obst_coord[:,1])).T
    
    # bottom
    bottom = np.array((obst_coord[:,0],-obst_coord[:,1])).T
    
    # top left
    tl = np.array((-obst_coord[:,0],2*y_length-obst_coord[:,1])).T
    
    # top right
    tr = np.array((2*x_length-obst_coord[:,0],2*y_length-obst_coord[:,1])).T
    
    # bottom left
    bl = np.array((-obst_coord[:,0],-obst_coord[:,1])).T
    
    # bottom right
    br = np.array((2*x_length-obst_coord[:,0],-obst_coord[:,1])).T
    
    og_obst_coord = np.copy(obst_coord) # only for plotting (debugging)
    obst_coord = np.vstack((obst_coord, top, left, right, bottom, tl, tr, bl, br))  

    
    # define the starting and end points (as random coordinates on the top and bottom boundary)
    start_coord = np.random.random_sample()*x_length
    end_coord = np.random.random_sample()*x_length

    ##### Step 2: Computing the Voronoi diagram
    # O(nlogn)
    vor = Voronoi(obst_coord, furthest_site=False)

    ##### Step 3: building the weighted graph
    # O(n)
    edges_w_weights, all_idx = compute_graph(vor, obst_coord, n_obst, x_length, y_length, start_coord, end_coord)

    ##### Step 4: Union-Find a connection from start to end
    graph, _ = union_find(edges_w_weights, all_idx)

    ##### Step 5: Find a path (doesn't matter how long as all of them are maximal distance to any cow)
    path, visited_nodes = find_path(graph, len(all_idx))

    print(time.time() - time_start, len(all_idx))
    # print("total runtime: ", time.time() - time_start, "[s]")

    ##### Step 6: Plot

    plot = False
    if plot:
        # Draw the square bounding box
        plt.plot([0, x_length, x_length, 0, 0],
                [0, 0, y_length, y_length, 0],
                'k--', lw=2)

        margin = max(x_length, y_length)/20.0
        plt.xlim(-margin, x_length + margin)
        plt.ylim(-margin, y_length + margin)

        # Plot the cows and the start/end points
        plt.scatter(og_obst_coord[:,0], og_obst_coord[:,1], c='pink', s=50)
        plt.plot(vor.vertices[all_idx[0], 0], vor.vertices[all_idx[0], 1], marker='x', linestyle='-', color='green', markersize=8)
        plt.plot(vor.vertices[all_idx[-1], 0], vor.vertices[all_idx[-1], 1], marker='x', linestyle='-', color='red', markersize=8)

        # Plot the found path
        x_coords = vor.vertices[all_idx[path], 0]
        y_coords = vor.vertices[all_idx[path], 1]
        plt.plot(x_coords, y_coords, marker='o', linestyle='-', color='blue', markersize=8)

        # plt.scatter(edges_w_weights[:,3], edges_w_weights[:,4], s=50, edgecolors='black')

        plt.xlabel('X Axis')
        plt.ylabel('Y Axis')
        plt.savefig('heatmap2.png')
        plt.show()
    

if __name__ == "__main__":
    main()
    