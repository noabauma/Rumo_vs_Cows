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
    
    # now we have to check if the middle_point m between the cows is actually on the vertex.
    # If not, the point (a or b) which is the closest to the cow will be cost of crossing the edge
    
    # step 1: check if m lies between
    
    # a + t*(b - a) = m if t in [0, 1] m is in between, if t in (-inf, 0) closer to a and if t in (1, inf) closer to b
    
    t = np.dot(b - a, m - a)/np.dot(b - a, b - a)
    
    if t < 0:
        cost = max(1 - np.linalg.norm(c1 - a)/crit_dist, 0.1)
    elif 0 <= t < 1:
        cost = max(1 - np.linalg.norm(c1 - m)/crit_dist, 0.1)
    else:
        cost = max(1 - np.linalg.norm(c1 - b)/crit_dist, 0.1)
    
    return cost, t
    
            

def main():
    """
    The main function to run the whole algorithm.
    It consists of:
    
    1. Building the Problem field
    2. Computing the special Voronoi Map
    3. Converting it into a weighted Graph
    4. Shortest Path Algorithm
    5. Plot
    """
    
    # Step 1: Let's build the problem field
    x_length = 4        # x coordinate of the cows field [m]
    y_length = 4        # y coordinate of the cows field [m]
    
    n_obst = 3          # number of obsticles (cows)
    obst_coord = np.random.rand(n_obst, 2)
    obst_coord[:,0] *= x_length
    obst_coord[:,1] *= y_length
    
    #obst_coord = np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]]) + 1
    #n_obst = len(obst_coord)          # number of obsticles (cows)
    
    # Mirroring
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
    
    og_obst_coord = np.copy(obst_coord)
    obst_coord = np.vstack((obst_coord, top, left, right, bottom, tl, tr, bl, br))  

    
    # define the starting and end points (as indices in the graph)
    start_coord = np.random.random_sample()*x_length
    end_coord = np.random.random_sample()*x_length

    
    # Step 2 & 3: Let's build the heat map via a self-made voronoi approach as we need to store the weights of the points
    vor = Voronoi(obst_coord, furthest_site=False)
    
    # delete the vertices that are outside the field
    eps = 1e-8
    
    keep_nodes = []
    for i in range(len(vor.vertices)):
        if vor.vertices[i, 0] >= -eps and vor.vertices[i, 0] <= x_length+eps and vor.vertices[i, 1] >= -eps and vor.vertices[i, 1] <= y_length+eps:
            keep_nodes.append(i)
    
    
    vor_nodes = np.copy(vor.vertices[keep_nodes])
    
    keep_edges = []
    for i in range(len(vor.ridge_vertices)):
        if vor.ridge_vertices[i][0] != -1:
            keep_edges.append(i)
    
    vor_edges = np.copy(np.array(vor.ridge_vertices)[keep_edges])
    
    
    # Draw the middle points of the ridges (also add weight and if on a infinit line)
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
    
    print(middle_points)
    
    # Next step: store everything into a weightes CSR graph file
    
    # First, make a mapping of the vor.vertices to arange as CSR starts from 0, n_points -1.
    all_idx = np.unique(middle_points[:,0:2])
    
    # TODO: define the end and starting point by shuffeling the all_idx! (amazing)
    # choose the one closest to the random sampled once
    
    
    n_nodes = len(all_idx)
    graph = np.zeros((n_nodes, n_nodes))
    for middle_point in middle_points:
        i = np.where(all_idx == middle_point[0])[0][0]
        j = np.where(all_idx == middle_point[1])[0][0]
        cost = middle_point[2]
        graph[i,j] = cost
        
    print(graph)
    
    graph = csr_matrix(graph)
    
    
    fig = voronoi_plot_2d(vor)
    margin = max(x_length, y_length)/20.0
    plt.xlim(-margin, x_length + margin)
    plt.ylim(-margin, y_length + margin)
    
    colors = ['green' if (0.0 < t < 1.0) else 'red' for t in middle_points[:,3]]
    
    plt.scatter(middle_points[:,4], middle_points[:,5], c=colors, s=50, edgecolors='black')
    
    plt.scatter(og_obst_coord[:,0], og_obst_coord[:,1], c='pink', s=50)
    
    plt.scatter(vor_nodes[:,0], vor_nodes[:,1], c='yellow', s=50, edgecolors='black')

    # Draw the square bounding box
    plt.plot([0, x_length, x_length, 0, 0],
            [0, 0, y_length, y_length, 0],
            'k--', lw=2)

    plt.savefig('heatmap2.png')
    plt.show()
    
    
    
     

if __name__ == "__main__":
    main()
    