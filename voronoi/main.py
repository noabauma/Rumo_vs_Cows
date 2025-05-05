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
    """_summary_

    Args:
        cow_coord (np.ndarray): _description_
        rumo_coord (np.ndarray): _description_
    """
    crit_dist = 10    # critical distance between cows [m] closer than this will lead to rumo barking
    
    return max(1 - np.linalg.norm(cow_coord - rumo_coord)/crit_dist, 0.0)
    
            

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
    x_length = 50        # x coordinate of the cows field [m]
    y_length = 50        # y coordinate of the cows field [m]
    n_obst = 10           # number of obsticles (cows)
    
    
    obst_coord = np.random.rand(n_obst, 2)
    
    # define the starting and end points (as indices in the graph)
    start_coord = "top"
    end_coord = "bottom"

    obst_coord[:,0] *= x_length
    obst_coord[:,1] *= y_length
    
    # Step 2 & 3: Let's build the heat map via a self-made voronoi approach as we need to store the weights of the points
    vor = Voronoi(obst_coord)
    
    print(vor.vertices)
    print(vor.ridge_vertices)
    
    # delete the vertices that are outside the field
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
    
    print(vor_nodes)
    print(vor_edges)
    
    fig = voronoi_plot_2d(vor, further_site=True)
    plt.savefig('heatmap2.png')
    plt.show()
    
    # # Step 4: Compute the shortest path
    # dist_matrix, predecessors = shortest_path(csgraph=graph, directed=False, indices=start_coord, return_predecessors=True)
    
    # # Backtrack to find the shortest path from source to destination
    # path = []
    # step = end_coord
    # while step != start_coord:
    #     path.append(step)
    #     step = predecessors[step]

    # path.append(start_coord)
    # path = path[::-1]  # Reverse the path to get it from source to destination
    
    # x_coords = grid_points[path, 0]
    # y_coords = grid_points[path, 1]
    
    
    # # Step 5: Plot
    # plt.plot(x_coords, y_coords, marker='o', linestyle='-', color='blue', markersize=8)
    
    # plt.plot(grid_points[start_coord, 0], grid_points[start_coord, 1], marker='x', linestyle='-', color='red', markersize=8)
    # plt.plot(grid_points[end_coord, 0], grid_points[end_coord, 1], marker='x', linestyle='-', color='red', markersize=8)

    # # Show the plot
    # weights = grid_points[:, 2].reshape((y_length+1, x_length+1))
    # plt.imshow(weights, origin='lower', extent=(0, x_length, 0, y_length), cmap='hot', interpolation='nearest')
    # plt.colorbar(label='Cost')
    
    
    # plt.title('Heatmap with Shortest Path')
    # plt.xlabel('X Axis')
    # plt.ylabel('Y Axis')
    # plt.show()
    
    
     

if __name__ == "__main__":
    main()
    