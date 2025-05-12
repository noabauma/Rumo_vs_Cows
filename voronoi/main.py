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


def cost_function(cow_coord: np.ndarray, rumo_coord: np.ndarray):
    return np.linalg.norm(cow_coord - rumo_coord)
    
            

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
    
    n_obst = 9          # number of obsticles (cows)
    obst_coord = np.random.rand(n_obst, 2)
    obst_coord[:,0] *= x_length
    obst_coord[:,1] *= y_length
    
    #obst_coord = np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]]) + 1
    #n_obst = len(obst_coord)          # number of obsticles (cows)
    
    # define the starting and end points (as indices in the graph)
    start_coord = 0
    end_coord = -1

    
    # Step 2 & 3: Let's build the heat map via a self-made voronoi approach as we need to store the weights of the points
    vor = Voronoi(obst_coord, furthest_site=False)
    
    print(vor.vertices)
    print(vor.ridge_vertices)
    
    # delete the vertices that are outside the field
    """
    keep_nodes = []
    for i in range(len(vor.vertices)):
        if vor.vertices[i, 0] >= 0 and vor.vertices[i, 0] <= x_length and vor.vertices[i, 1] >= 0 and vor.vertices[i, 1] <= y_length:
            keep_nodes.append(i)
    
    print(keep_nodes)
    
    vor_nodes = np.copy(vor.vertices[keep_nodes])
    
    keep_edges = []
    for i in range(len(vor.ridge_vertices)):
        if vor.ridge_vertices[i][0] != -1:
            keep_edges.append(i)
    
    vor_edges = np.copy(np.array(vor.ridge_vertices)[keep_edges])
    """
    
    # Draw the middle points of the ridges (also add weight and if on a infinit line)
    middle_points = np.empty((len(vor.ridge_points), 4))
    for i, ridge_point in enumerate(vor.ridge_points):
        middle_points[i, 0:2] = np.array(obst_coord[ridge_point[0]] + obst_coord[ridge_point[1]])/2
        middle_points[i, 2] = cost_function(obst_coord[ridge_point[0]], obst_coord[ridge_point[1]])
        middle_points[i, 3] = vor.ridge_vertices[i][0] != -1
        
    print(middle_points)
    
    fig = voronoi_plot_2d(vor)
    plt.xlim(0, x_length)
    plt.ylim(0, y_length)
    
    colors = ['green' if label else 'red' for label in middle_points[:,3]]
    
    plt.scatter(middle_points[:,0], middle_points[:,1], c=colors, s=50, edgecolors='black')

    # Draw the square bounding box
    plt.plot([0, x_length, x_length, 0, 0],
            [0, 0, y_length, y_length, 0],
            'k--', lw=2)

    plt.savefig('heatmap2.png')
    plt.show()
    
    
    
     

if __name__ == "__main__":
    main()
    