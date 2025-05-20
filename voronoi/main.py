import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

"""
This code computes the problem of Rumo having to pass a field with n cows.
We have to get from point A to point B with having as little contact to the cows as he start barking otherwise.
I.e. we are searching for the path of least resistance.
"""

np.random.seed(42)


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
        cost = max(1 - np.linalg.norm(c1 - m)/crit_dist, 0.1)
        #cost = max(1/np.dot(c1 - m, c1 - m), 0.1)
    elif t < 0:
        cost = max(1 - np.linalg.norm(c1 - a)/crit_dist, 0.1)
        #cost = max(1/np.dot(c1 - a, c1 - a), 0.1)
    else:
        cost = max(1 - np.linalg.norm(c1 - b)/crit_dist, 0.1)
        #cost = max(1/np.dot(c1 - b, c1 - b), 0.1)
    
    return cost, t
    
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
    middle_points = []
    for i, ridge_point in enumerate(vor.ridge_points):        
        # go through all the ridge_points with atleast one being inside the boundary
        if not (ridge_point[0] >= n_obst and ridge_point[1] >= n_obst):
            assert vor.ridge_vertices[i][0] != -1, "Somehow, this vertex has only one voronoi node. Should not happen with Jonah's mirroring technique!"
            
            middle_point = np.empty((6))

            middle_point[0:2] = vor.ridge_vertices[i]
            
            middle_point[2:4] = cost_function(vor.vertices[vor.ridge_vertices[i][0]], vor.vertices[vor.ridge_vertices[i][1]], obst_coord[ridge_point[0]], obst_coord[ridge_point[1]])
            middle_point[4:6] = np.array(obst_coord[ridge_point[0]] + obst_coord[ridge_point[1]])/2
            
            middle_points.append(middle_point)
            
            
    middle_points = np.array(middle_points)
    
    #print(middle_points)
    
    # Next step: store everything into a weightes CSR graph file
    
    # First, make a mapping of the vor.vertices to arange as CSR starts from 0, n_points -1.
    all_idx = np.unique(middle_points[:,0:2]).astype(int)
    
    # We define the end and starting point by shuffeling the all_idx! (amazing)
    # The first index is the starting point and the last index the end point
    # The starting point will start on a point on the lower boundary
    # and the end point on a point on the upper boundary
    # We choose the start/endpoints which are the closest to the ridge point on the respective boundaries
    closest_to_start = (-1, x_length)
    closest_to_end = (-1, x_length)
    for idx in all_idx:
        vor_vertex = vor.vertices[idx]
        if vor_vertex[1] == 0.:
            dist = abs(vor_vertex[0] - start_coord)
            
            if dist < closest_to_start[1]:
                closest_to_start = (idx, dist)
                
        elif vor_vertex[1] == y_length:
            dist = abs(vor_vertex[0] - end_coord)
            
            if dist < closest_to_end[1]:
                closest_to_end = (idx, dist)
                
    assert closest_to_start[0] != -1, "didn't find a closest point on the lower boundary"
    assert closest_to_end[0] != -1, "didn't find a closest point on the upper boundary"   
    
    start_idx = np.where(all_idx == closest_to_start[0])[0][0]
    end_idx = np.where(all_idx == closest_to_end[0])[0][0]  
                
    all_idx[0], all_idx[start_idx] = all_idx[start_idx], all_idx[0]
    all_idx[-1], all_idx[end_idx] = all_idx[end_idx], all_idx[-1]
    
    n_nodes = len(all_idx)
    graph = np.zeros((n_nodes, n_nodes))
    for middle_point in middle_points:
        i = np.where(all_idx == middle_point[0])[0][0]
        j = np.where(all_idx == middle_point[1])[0][0]
        cost = middle_point[2]
        graph[i,j] = cost
    
    return csr_matrix(graph), all_idx
            

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
    
    x_length = 50        # x coordinate of the cows field [m]
    y_length = 100        # y coordinate of the cows field [m]
    n_obst = 100          # number of obsticles (cows)
    
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

    
    ##### Step 2: Computing the Voronoi diagram
    # O(nlogn)
    vor = Voronoi(obst_coord, furthest_site=False)
    
    ##### Step 3: building the weighted graph
    # O(n)
    graph, all_idx = compute_graph(vor, obst_coord, n_obst, x_length, y_length, start_coord, end_coord)
    
    ##### Step 4: Compute the shortest path
    # O[n*(n*k + n*log(n))] with k in [3,6]
    dist_matrix, predecessors = shortest_path(csgraph=graph, method='auto', directed=False, indices=0, return_predecessors=True)
    
    # Backtrack to find the shortest path from source to destination
    path = []
    step = -1
    while step != 0:
        path.append(step)
        step = predecessors[step]

    path.append(0)
    path = path[::-1]  # Reverse the path to get it from source to destination
    
    # total runtime complexity
    # 
    print("total runtime: ", time.time() - time_start, "[s]")
    
    ##### Step 5: Plot
    
    # plot voronoi
    #fig = voronoi_plot_2d(vor)
    
    # Plot the shortest path    
    x_coords = vor.vertices[all_idx[path], 0]
    y_coords = vor.vertices[all_idx[path], 1]
    plt.plot(x_coords, y_coords, marker='o', linestyle='-', color='blue', markersize=8)
    
    #colors = ['green' if (0.0 < t < 1.0) else 'red' for t in middle_points[:,3]]
    #plt.scatter(middle_points[:,4], middle_points[:,5], c=colors, s=50, edgecolors='black')
    
    plt.scatter(og_obst_coord[:,0], og_obst_coord[:,1], c='pink', s=50)

    # Draw the square bounding box
    plt.plot([0, x_length, x_length, 0, 0],
            [0, 0, y_length, y_length, 0],
            'k--', lw=2)
    
    margin = max(x_length, y_length)/20.0
    plt.xlim(-margin, x_length + margin)
    plt.ylim(-margin, y_length + margin)

    plt.title('Heatmap with Shortest Path')
    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')
    plt.savefig('heatmap2.png')
    plt.show()
    

if __name__ == "__main__":
    main()
    