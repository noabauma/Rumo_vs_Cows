import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

"""
This code computes the problem of Rumo having to pass a field with n cows.
We have to get from point A to point B with having as little contact with the cows as he start barking otherwise.
I.e. we are searching for the path of least resistance.
"""


def cost_function(cow_coord: np.ndarray, rumo_coord: np.ndarray):
    """Cost function to compute the cost to stay on this given node to a cow.

    Args:
        cow_coord (np.ndarray): the 2d coordinate of a cow.
        rumo_coord (np.ndarray): the current 2d coordinate of rumo.
    """
    crit_dist = 10    # critical distance between cows [m] closer than this will lead to rumo barking
    
    return max(1 - np.linalg.norm(cow_coord - rumo_coord)/crit_dist, 0.1)

def swap_nodes(G, i, j):
    """Swap nodes i and j in the csr graph

    Args:
        G (csr_matrix): The graph
        i (int): The index of the first node
        j (int): The index of the second node

    Returns:
        csr_matrix: The swapped graph
    """
    # Convert to LIL format for easier row and column manipulation
    G = G.tolil()

    # Swap rows i and j
    G[[i, j], :] = G[[j, i], :]

    # Swap columns i and j
    G[:, [i, j]] = G[:, [j, i]]

    return G.tocsr()
    


def compute_heatmap(obst_coord: np.ndarray, x_length: float = 1, y_length: float = 1, grid_spacing: float = 1):
    """This function computes the heatmap of the cows as a graph.
        The algorithm computes a head map of how close rumo is allowed to come.
        Graph is connected by the 8 neighbours.

    Args:
        obst_coord (np.ndarray): The 2d coordinates of all the cows
        x_length (float, optional): The x length of the field. Defaults to 1.
        y_length (float, optional): The y length of the field. Defaults to 1.
        grid_spacing (float, optional): The spacing of the grid points [m]. Defaults to 1.

    Returns:
        np.ndarray: The grid points with their weights shape (n_total_points, 3)
        csr matrix: The graph of shape (n_total_points*8, 3)
    """
    n_grid_points_x = int(x_length/grid_spacing + 1)    # number of grid points in the x-dimension
    n_grid_points_y = int(y_length/grid_spacing + 1)    # number of grid points in the y-dimension
    n_total_points = n_grid_points_x*n_grid_points_y
    
    # Step 1: Generate equally spaced points between in the x and y dimensions
    x_points = np.linspace(0, x_length, n_grid_points_x)
    y_points = np.linspace(0, y_length, n_grid_points_y)
    
    # Step 2: Create the 2D grid using meshgrid
    x, y = np.meshgrid(x_points, y_points)
    
    # Step 3: Combine the grid coordinates into an array of points
    grid_points = np.vstack([x.ravel(), y.ravel()]).T
    
    # add another column for the weights
    grid_points = np.hstack([grid_points, np.zeros((n_total_points, 1))])
    
    # compute the weights
    for grid_point in grid_points:
        cost = 0
        for cow_coord in obst_coord:
            cost = max(cost, cost_function(cow_coord, grid_point[:2]))
        
        # store the cost
        grid_point[2] = cost
        
    return grid_points, n_total_points, n_grid_points_x

def compute_graph(grid_points: np.array, n_total_points: int, n_grid_points_x: int):
    """AI is creating summary for compute_graph

    Args:
        grid_points (np.array): [description]
        n_total_points (int): [description]
        n_grid_points_x (int): [description]

    Returns:
        [type]: [description]
    """
    
    # create the dense matrix for the graph (zero means no connection)
    graph = np.zeros((n_total_points, n_total_points))
    
    # now transform node weights into edge weights
    # nodes are connected by the 8 neighbours
    # we have to be careful with the boundaries as we don't want to go out of bounds.
    # 
    #   y
    #   ^
    #   |
    #   |
    #   +----> x
    #
    for i in range(n_total_points):
        
        # left boundary
        if i % n_grid_points_x != 0:
            graph[i, i-1] = (grid_points[i, 2] + grid_points[i-1, 2])/2
            
        # right boundary
        if i % n_grid_points_x != n_grid_points_x-1:
            graph[i, i+1] = (grid_points[i, 2] + grid_points[i+1, 2])/2
        
        # top boundary
        if i < n_total_points - n_grid_points_x:
            graph[i, i+n_grid_points_x] = (grid_points[i, 2] + grid_points[i+n_grid_points_x, 2])/2
            
            # top left corner
            if i % n_grid_points_x != 0:
                graph[i, i+n_grid_points_x-1] = (grid_points[i, 2] + grid_points[i+n_grid_points_x-1, 2])/2
                
            # top right corner
            if i % n_grid_points_x != n_grid_points_x-1:
                graph[i, i+n_grid_points_x+1] = (grid_points[i, 2] + grid_points[i+n_grid_points_x+1, 2])/2
        
        # bottom boundary
        if i > n_grid_points_x-1:
            graph[i, i-n_grid_points_x] = (grid_points[i, 2] + grid_points[i-n_grid_points_x, 2])/2
            
            # bottom left corner
            if i % n_grid_points_x != 0:
                graph[i, i-n_grid_points_x-1] = (grid_points[i, 2] + grid_points[i-n_grid_points_x-1, 2])/2
            
            # bottom right corner
            if i % n_grid_points_x != n_grid_points_x-1:
                graph[i, i-n_grid_points_x+1] = (grid_points[i, 2] + grid_points[i-n_grid_points_x+1, 2])/2

    # convert the graph into CSR format (for efficiency)
    graph = csr_matrix(graph)

    return graph
            

def main():
    """
    The main function to run the whole algorithm.
    It consists of:
    
    1. Building the Problem field
    2. Computing the 2d heatmap of the cows in the field
    3. Converting the heatmap into a weighted graph
    4. Compute shortest path
    5. Plot
    """
    
    time_start = time.time()
    
    ##### Step 1: Let's build the problem field
    x_length = 50        # x coordinate of the cows field [m]
    y_length = 100        # y coordinate of the cows field [m]
    n_obst = 100          # number of obsticles (cows)
    
    np.random.seed(42)   # seed for the random number generator
        
    grid_spacing = 1        # spacing of the grid points [m]
    
    obst_coord = np.random.rand(n_obst, 2)
    
    obst_coord[:,0] *= x_length
    obst_coord[:,1] *= y_length

    ##### Step 2: Computing the 2d heatmap of the cows in the field
    grid_points, n_total_points, n_grid_points_x = compute_heatmap(obst_coord, x_length, y_length, grid_spacing)
    
    ##### Step 3: Converting the heatmap into a weighted graph
    graph = compute_graph(grid_points, n_total_points, n_grid_points_x)
    
    # Define the starting and end points (as indices in the graph)    
    start_coord = int(np.random.random_sample()*x_length + 0.5)                                                         # 0
    end_coord = grid_points.shape[0] - int(x_length/grid_spacing + 1) + int(np.random.random_sample()*x_length + 0.5)   # -1
    
    # Swap the end point with the current last one
    swap_nodes(graph, end_coord, -1)
    
    ##### Step 4: Compute the shortest path
    dist_matrix, predecessors = shortest_path(csgraph=graph, method='auto', directed=False, indices=start_coord, return_predecessors=True)
    
    # Backtrack to find the shortest path from source to destination
    path = []
    step = end_coord
    while step != start_coord:
        path.append(step)
        step = predecessors[step]

    path.append(start_coord)
    path = path[::-1]  # Reverse the path to get it from source to destination
    
    print("total runtime: ", time.time() - time_start, "[s]")
    
    ##### Step 5: Plot
    
    # Plot the shortest path
    x_coords = grid_points[path, 0]
    y_coords = grid_points[path, 1]
    plt.plot(x_coords, y_coords, marker='o', linestyle='-', color='blue', markersize=8)
    
    # Plot the start and end points
    #plt.plot(grid_points[start_coord, 0], grid_points[start_coord, 1], marker='x', linestyle='-', color='green', markersize=8)
    #plt.plot(grid_points[end_coord, 0], grid_points[end_coord, 1], marker='x', linestyle='-', color='red', markersize=8)

    # Plot the heatmap
    weights = grid_points[:, 2].reshape((y_length+1, x_length+1))
    plt.imshow(weights, origin='lower', extent=(0, x_length, 0, y_length), cmap='hot', interpolation='nearest')
    plt.colorbar(label='Cost')
    
    margin = max(x_length, y_length)/20.0
    plt.xlim(-margin, x_length + margin)
    plt.ylim(-margin, y_length + margin)
    
    plt.title('Heatmap with Shortest Path')
    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')
    plt.savefig('heatmap.png')
    plt.show()
    
    
     

if __name__ == "__main__":
    main()
    