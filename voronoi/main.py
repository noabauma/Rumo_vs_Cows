import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.integrate import simpson

"""
This code computes the problem of Rumo having to pass a field with n cows.
We have to get from point A to point B with as little contact with the cows as possible, as he starts barking otherwise.
I.e. we are searching for the path of least resistance.
"""


def cost_function(a: np.ndarray, b: np.ndarray, c1: np.ndarray, c2: np.ndarray):
    """Cost function to determine the cost of crossing this edge

    Args:
        a (np.ndarray): Starting coordinates in 2d
        b (np.ndarray): End coordinate in 2d
        c1 (np.ndarray): cow 1 coordinate in 2d
        c2 (np.ndarray): cow 2 coordinate in 2d

    Returns:
        cost (float): The cost of crossing this edge
    """

    # Generate n evenly spaced points along the edge
    n = 10
    ts = np.linspace(0, 1, n)
    points = a[None, :] + ts[:, None] * (b - a)[None, :]  # shape (n, 2)

    # Compute cost at each sampled point.
    # Every point on a voronoi ridge is equidistant from its two generating cows,
    # so the distance to c1 is already the distance to the nearest cow (c2 would give the same).
    crit_dist = 10
    costs = np.maximum(1 - np.linalg.norm(points - c1, axis=1) / crit_dist, 0.1)

    # Scale by arc length of the edge (distance from a to b)
    dist = np.linalg.norm(b - a)
    cost = simpson(costs, ts) * dist
    
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
        csr_matrix: The weighted graph
        np.ndarray: mapping from graph node indices to voronoi vertex indices
                    (the start node comes first, the end node last)
    """

    # We start with computing all the middle points of the ridges (also add weight and other important stuff)
    # one middle_point consists of: [start_idx, end_idx, cost, x_coord, y_coord]
    # start_idx: starting ridge_point idx
    # end_idx: ending ridge_point idx
    # cost: the cost of crossing this edge
    # x_coord: of the middle point between the two cows (debugging purpose)
    # y_coord: of the middle point between the two cows (debugging purpose)
    edges_w_weights = []
    for i, ridge_point in enumerate(vor.ridge_points):        
        # go through all the ridge_points inside and on the rectangle field
        if not (ridge_point[0] >= n_obst and ridge_point[1] >= n_obst):
            assert vor.ridge_vertices[i][0] != -1, "Somehow, this vertex has only one voronoi node. Should not happen with Jonah's mirroring technique!"
            
            middle_point = np.empty((5))

            middle_point[0:2] = vor.ridge_vertices[i]
            
            middle_point[2] = cost_function(vor.vertices[vor.ridge_vertices[i][0]], vor.vertices[vor.ridge_vertices[i][1]], obst_coord[ridge_point[0]], obst_coord[ridge_point[1]])
            middle_point[3:] = np.array(obst_coord[ridge_point[0]] + obst_coord[ridge_point[1]])/2
            
            edges_w_weights.append(middle_point)
            
            
    edges_w_weights = np.array(edges_w_weights)

    # Next step: store everything into a weighted CSR graph

    # First, make a mapping of the vor.vertices to arange as CSR starts from 0, n_points -1.
    all_idx = np.unique(edges_w_weights[:,0:2]).astype(int)

    # We define the end and starting point by swapping the first and last position in all_idx! (amazing)
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

    # We create a mapping between indices {0, n-1} and the real all_idx for O(1) lookup time
    idx_map = {val: idx for idx, val in enumerate(all_idx)}

    n_nodes = len(all_idx)

    # build the graph directly in sparse (COO) format instead of filling a dense n_nodes x n_nodes matrix
    rows = [idx_map[int(edge[0])] for edge in edges_w_weights]
    cols = [idx_map[int(edge[1])] for edge in edges_w_weights]
    graph = csr_matrix((edges_w_weights[:, 2], (rows, cols)), shape=(n_nodes, n_nodes))

    return graph, all_idx
            

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
    graph, all_idx = compute_graph(vor, obst_coord, n_obst, x_length, y_length, start_coord, end_coord)
    
    ##### Step 4: Compute the shortest path
    # O[n*k + n*log(n)] with k in [3,6] -> O(n*log(n))
    dist_matrix, predecessors = shortest_path(csgraph=graph, method='D', directed=False, indices=0, return_predecessors=True)
    
    # Backtrack to find the shortest path from source to destination
    path = []
    step = -1
    while step != 0:
        path.append(step)
        step = predecessors[step]

    path.append(0)
    path = path[::-1]  # Reverse the path to get it from source to destination
    
    # total runtime complexity
    # O[n*log(n) + n + n*k + n*log(n)]
    print(time.time() - time_start, len(all_idx))
    # print("total runtime: ", time.time() - time_start, "[s]")
    
    ##### Step 5: Plot
    
    plot = False
    if plot:
        # Plot the shortest path
        x_coords = vor.vertices[all_idx[path], 0]
        y_coords = vor.vertices[all_idx[path], 1]
        plt.plot(x_coords, y_coords, marker='o', linestyle='-', color='blue', markersize=8)
        
        #colors = ['green' if (0.0 < t < 1.0) else 'red' for t in edges_w_weights[:,3]]
        #plt.scatter(edges_w_weights[:,4], edges_w_weights[:,5], c=colors, s=50, edgecolors='black')
        
        plt.scatter(og_obst_coord[:,0], og_obst_coord[:,1], c='pink', s=50)

        # Draw the square bounding box
        plt.plot([0, x_length, x_length, 0, 0],
                [0, 0, y_length, y_length, 0],
                'k--', lw=2)
        
        margin = max(x_length, y_length)/20.0
        plt.xlim(-margin, x_length + margin)
        plt.ylim(-margin, y_length + margin)

        plt.title('Shortest Path in a Voronoi Diagram of Cows')
        plt.xlabel('X Axis')
        plt.ylabel('Y Axis')
        plt.savefig('heatmap2.png')
        plt.show()
    

if __name__ == "__main__":
    main()
    