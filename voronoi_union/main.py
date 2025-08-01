import sys
import time
import numpy as np
from collections import defaultdict, deque
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.sparse import csr_matrix
from scipy.integrate import simpson

"""
This code computes the problem of Rumo having to pass a field with n cows.
We have to get from point A to point B with having as little contact to the cows as he start barking otherwise.
I.e. we are searching for the path of least resistance.
"""

# Turn on interactive mode
# plt.ion()
# plt.switch_backend('Agg')  # Offscreen rendering

# Create a figure and axis once
fig, ax = plt.subplots()


class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))

    def find(self, i):
        # If i itself is root or representative
        if self.parent[i] == i:
            return i
          
        # Else recursively find the representative 
        # of the parent
        return self.find(self.parent[i])

    def union(self, i, j):
        ir, jr = self.find(i), self.find(j)
        if ir == jr:
            return False
        
        self.parent[ir] = jr

        return True


def cost_function(a: np.array, b: np.array, c1: np.array, c2: np.array):
    """Cost function to determine the cost of crossing this edge

    Args:
        a (np.array): Starting coordinates in 2d
        b (np.array): End coordinate in 2d
        c1 (np.array): cow 1 coordinate in 2d
        c2 (np.array): cow 2 coordinate in 2d

    Returns:
        cost (float): The cost of crossing this edge
    """
    crit_dist = 5    # critical distance between cows [m] closer than this will lead to rumo barking
    
    m = (c1 + c2)/2 # middle_point
    
    # now we have to check if the middle_point m between is actually on the vertex.
    # If not, the point (a or b) which is the closest to the cow will be the cost of crossing the edge

    # If t in [0, 1]: m is in between, if t in (-inf, 0): m closer to a and if t in (1, inf): closer to b
    
    t = np.dot(b - a, m - a)/np.dot(b - a, b - a)
    
    if 0 <= t <= 1:
        cost = np.linalg.norm(c1 - m)
    elif t < 0:
        cost = np.linalg.norm(c1 - a)
    else:
        cost = np.linalg.norm(c1 - b)
    
    return cost
    
def compute_graph(vor: Voronoi, obst_coord: np.array, n_obst: int, x_length: float, y_length: float, start_coord: float, end_coord: float):
    """Computing the weighted graph from the voronoi diagram

    Args:
        vor (Voronoi): [description]
        obst_coord (np.array): [description]
        n_obst (int): [description]
        x_length (float): [description]
        y_length (float): [description]
        start_coord (float): [description]
        end_coord (float): [description]
    """
    
    # We start with computing all the middle points of the ridges (also add weight and other important stuff)
    # one middle_point consists of: [start_idx, end_idx, cost, t, x_coord, y_coord]
    # start_idx: starting ridge_point idx
    # end_idx: ending ridge_point idx
    # cost: the cost of crossing this edge
    # t: if t in [0, 1] the middle point lies between the ridge points else it is outside (debugging purpose)
    # x_coord: of the middle_point (debugging purpose)
    # y_coord: of the middle_point (debugging purpose)
    edges_w_weights = []
    for i, ridge_point in enumerate(vor.ridge_points):        
        # go through all the ridge_points with atleast one being inside the boundary
        if not (ridge_point[0] >= n_obst and ridge_point[1] >= n_obst):
            assert vor.ridge_vertices[i][0] != -1, "Somehow, this vertex has only one voronoi node. Should not happen with Jonah's mirroring technique!"
            
            middle_point = np.empty((5))

            middle_point[0:2] = vor.ridge_vertices[i]
            
            middle_point[2] = cost_function(vor.vertices[vor.ridge_vertices[i][0]], vor.vertices[vor.ridge_vertices[i][1]], obst_coord[ridge_point[0]], obst_coord[ridge_point[1]])
            middle_point[3:] = np.array(obst_coord[ridge_point[0]] + obst_coord[ridge_point[1]])/2
            
            edges_w_weights.append(middle_point)
            
            
    edges_w_weights = np.array(edges_w_weights)
    
    # Sort by column by best graph (i.e. highest score)
    edges_w_weights = edges_w_weights[np.argsort(edges_w_weights[:, 2])[::-1]]
    
    # First, make a mapping of the vor.vertices to arange as CSR starts from 0, n_points -1.
    all_idx = np.unique(edges_w_weights[:,0:2]).astype(int)
    
    # We define the end and starting point by shuffeling the all_idx! (amazing)
    # The first index is the starting point and the last index the end point
    # The starting point will start on a point on the lower boundary
    # and the end point on a point on the upper boundary
    # We choose the start/endpoints which are the closest to the ridge point on the respective boundaries
    closest_to_start = (-1, x_length)
    closest_to_end = (-1, x_length)
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
    all_idx[-1], all_idx[end_idx] = all_idx[end_idx], all_idx[-1]
    
    return edges_w_weights, all_idx

    


def union_find(vor: Voronoi, edges_w_weights: np.array, all_idx: np.array):
    """Union Find algorithm in finding the optimal path

    Args:
        edges_w_weights (np.array): Array of the x,y coordinate as well as the cost of each edge
        all_idx (np.array): the actual node indeces of the voronoi mesh
    """
    # We create a mapping between indeces {0, n-1} and the real all_idx for O(1) lookup time
    idx_map = {val: idx for idx, val in enumerate(all_idx)}
    
    n_nodes = len(all_idx)
    uf = UnionFind(n_nodes)
    graph = defaultdict(list)

    n_edges = edges_w_weights.shape[0]
    edge_idx = 0
    while uf.find(0) != uf.find(n_nodes - 1):
        
        i = int(edges_w_weights[edge_idx, 0])
        j = int(edges_w_weights[edge_idx, 1])
        
        # x_coords = [vor.vertices[i, 0], vor.vertices[j, 0]]
        # y_coords = [vor.vertices[i, 1], vor.vertices[j, 1]]
        # ax.plot(x_coords, y_coords, marker='o', linestyle='-', color='red', markersize=8)
        # plt.draw()     # Redraw the current figure
        # plt.pause(0.1) # Pause to allow the plot to update (simulate "step-by-step")
        # time.sleep(0.05)
        # fig.savefig(f"frame_{edge_idx:03d}.png")
        
        i = idx_map[i]
        j = idx_map[j]
        
        uf.union(i, j)
            
        graph[i].append(j)
        graph[j].append(i)

        edge_idx += 1
        
        assert edge_idx != n_edges, f"We did not find a path! (Impossible?!) {edge_idx} == {n_edges}"
    
    return graph, edge_idx

def dfs(graph: defaultdict, visited: list, path: list, node: int, n_nodes: int):
    path = path + [node]  # create a new list to avoid side effects
    visited.append(node)

    if node == n_nodes - 1:
        return path  # found target

    for neighbor in graph[node]:
        if neighbor not in visited:
            result = dfs(graph, visited, path, neighbor, n_nodes)
            if result:  # if a path was found
                return result

    return None  # no path found in this branch

def find_path(graph: defaultdict, n_nodes: int, least_visited: bool = False):
    """

    Args:
        graph (defaultdict): [description]
        n_nodes (int): [description]
    """
    if n_nodes > 6000:
        # Non-recursive DFS from ChatGPT
        visited = set()
        stack = [(0, [0])]  # (current_node, path_so_far)

        while stack:
            node, path = stack.pop()
            if node == n_nodes - 1:
                break  # Found target node
            for neighbor in graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)  # Mark as visited when pushing
                    stack.append((neighbor, path + [neighbor]))
        
        return path, visited
    else:
        # Recrsive DFS from myself (only works up to ~6000 cows)
        start_node = 0
        visited = [start_node] # make it to a list because of order perservation (i.e. manim)
        for neighbor in graph[start_node]:
            path = dfs(graph, visited, [start_node], neighbor, n_nodes)
            
            if path:
                break
            
        assert path, "Did not find a path"
            
        return path, visited
    
    
def main():
    """
    The main function to run the whole algorithm.
    It consists of:
    
    1. Defining the problem field (#cows, dimensions, start/end point, ...)
    2. Computing the Voronoi diagram
    3. Computing the cost of the edges and store it as a weighted graph
    4. Computing the shortest path
    5. Plot
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
        x_length = 50        # x coordinate of the cows field [m]
        y_length = 100        # y coordinate of the cows field [m]
        n_obst = 100          # number of obsticles (cows)
        
        np.random.seed(43)   # seed for the random number generator
    
    obst_coord = np.random.rand(n_obst, 2) # 2d coordinates of the cows
    obst_coord[:,0] *= x_length
    obst_coord[:,1] *= y_length
    
    # Mirroring the cow field as we also need the voronoi edge on the on the boundaries
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
    
    # Draw the square bounding box
    plt.plot([0, x_length, x_length, 0, 0],
            [0, 0, y_length, y_length, 0],
            'k--', lw=2)
    
    margin = max(x_length, y_length)/20.0
    plt.xlim(-margin, x_length + margin)
    plt.ylim(-margin, y_length + margin)
    
    plt.scatter(og_obst_coord[:,0], og_obst_coord[:,1], c='pink', s=50)
    
    ##### Step 2: Computing the Voronoi diagram
    # O(nlogn)
    vor = Voronoi(obst_coord, furthest_site=False)
    
    ##### Step 3: building the weighted graph
    # O(n)
    edges_w_weights, all_idx = compute_graph(vor, obst_coord, n_obst, x_length, y_length, start_coord, end_coord)
    
    plt.plot(vor.vertices[all_idx[0], 0], vor.vertices[all_idx[0], 1], marker='x', linestyle='-', color='green', markersize=8)
    plt.plot(vor.vertices[all_idx[-1], 0], vor.vertices[all_idx[-1], 1], marker='x', linestyle='-', color='red', markersize=8)
    
    ##### Step 4: Union-Find a connection from start to end
    graph, _ = union_find(vor, edges_w_weights, all_idx)
    
    ##### Step 5: Find a path (doesn't matter how long as all of them are maximal distance to any cow)
    path, visited_nodes = find_path(graph, len(all_idx))
    
    # total runtime complexity
    # 
    
    print(time.time() - time_start)
    # print("total runtime: ", time.time() - time_start, "[s]")
    
    ##### Step 6: Plot
    
    plot = False
    if plot:
        # plot voronoi
        #fig = voronoi_plot_2d(vor)
        
        # Plot the shortest path    
        x_coords = vor.vertices[all_idx[path], 0]
        y_coords = vor.vertices[all_idx[path], 1]
        plt.plot(x_coords, y_coords, marker='o', linestyle='-', color='blue', markersize=8)
        
        #plt.scatter(edges_w_weights[:,3], edges_w_weights[:,4], s=50, edgecolors='black')

        plt.xlabel('X Axis')
        plt.ylabel('Y Axis')
        plt.savefig('heatmap2.png')
        plt.show()
    

if __name__ == "__main__":
    main()
    