import numpy as np
import matplotlib.pyplot as plt
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
    
    return max(1 - np.linalg.norm(cow_coord - rumo_coord)/crit_dist, 0.1)
    


def build_voronoi_graph(obst_coord: np.ndarray, x_length: float = 1, y_length: float = 1) -> np.ndarray:
    """This function computes the 2D Voronoi Map of the cows field in an discrete way.
        Thew algorithm computes a head map of how close rumo is allowed to come.
        Graph is connected by the 8 neighbours.

    Args:
        obst_coord (np.ndarray): [description]
        x_length (float, optional): [description]. Defaults to 1.
        y_length (float, optional): [description]. Defaults to 1.

    Returns:
        np.ndarray: The grid points with their weights shape (n_total_points, 3)
        np.ndarray: The graph shape (n_total_points, n_total_points)
    """
    point_spacing = 1      # spacing of the grid points [m]
    n_grid_points_x = int(x_length/point_spacing + 1)    # number of grid points in the x-dimension
    n_grid_points_y = int(y_length/point_spacing + 1)    # number of grid points in the y-dimension
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

    
    # create the dense matrix for the graph (zero means no connection)
    graph = np.zeros((n_total_points, n_total_points))
    
    # now transform node weights into edge weights
    # nodes are connected by the 8 neighbours
    # we have to be careful with the boundary conditions
    # 
    #   y
    #   ^
    #   |
    #   |
    #   +----> x
    #
    for i in range(n_total_points):
        
        # left
        if i % n_grid_points_x != 0:
            graph[i, i-1] = (grid_points[i, 2] + grid_points[i-1, 2])/2
            
        # right
        if i % n_grid_points_x != n_grid_points_x-1:
            graph[i, i+1] = (grid_points[i, 2] + grid_points[i+1, 2])/2
        
        # top
        if i < n_total_points - n_grid_points_x:
            graph[i, i+n_grid_points_x] = (grid_points[i, 2] + grid_points[i+n_grid_points_x, 2])/2
            
            # top left
            if i % n_grid_points_x != 0:
                graph[i, i+n_grid_points_x-1] = (grid_points[i, 2] + grid_points[i+n_grid_points_x-1, 2])/2
                
            # top right
            if i % n_grid_points_x != n_grid_points_x-1:
                graph[i, i+n_grid_points_x+1] = (grid_points[i, 2] + grid_points[i+n_grid_points_x+1, 2])/2
        
        # bottom
        if i > n_grid_points_x-1:
            graph[i, i-n_grid_points_x] = (grid_points[i, 2] + grid_points[i-n_grid_points_x, 2])/2
            
            # bottom left
            if i % n_grid_points_x != 0:
                graph[i, i-n_grid_points_x-1] = (grid_points[i, 2] + grid_points[i-n_grid_points_x-1, 2])/2
            
            # bottom right
            if i % n_grid_points_x != n_grid_points_x-1:
                graph[i, i-n_grid_points_x+1] = (grid_points[i, 2] + grid_points[i-n_grid_points_x+1, 2])/2

    # convert the graph into a sparse matrix (for efficiency)
    graph = csr_matrix(graph)

    return grid_points, graph
            

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
    x_length = 75        # x coordinate of the cows field [m]
    y_length = 75        # y coordinate of the cows field [m]
    n_obst = 10           # number of obsticles (cows)
    
    
    obst_coord = np.random.rand(n_obst, 2)
    
    obst_coord[:,0] *= x_length
    obst_coord[:,1] *= y_length

    # Step 2 & 3: Let's build the heat map via a self-made voronoi approach as we need to store the weights of the points
    grid_points, graph = build_voronoi_graph(obst_coord, x_length, y_length)
    
    # define the starting and end points (as indices in the graph)
    start_coord = 0
    end_coord = -1
    
    # Step 4: Compute the shortest path
    dist_matrix, predecessors = shortest_path(csgraph=graph, directed=False, indices=start_coord, return_predecessors=True)
    
    # Backtrack to find the shortest path from source to destination
    path = []
    step = end_coord
    while step != start_coord:
        path.append(step)
        step = predecessors[step]

    path.append(start_coord)
    path = path[::-1]  # Reverse the path to get it from source to destination
    
    x_coords = grid_points[path, 0]
    y_coords = grid_points[path, 1]
    
    
    # Step 5: Plot
    plt.plot(x_coords, y_coords, marker='o', linestyle='-', color='blue', markersize=8)
    
    plt.plot(grid_points[start_coord, 0], grid_points[start_coord, 1], marker='x', linestyle='-', color='red', markersize=8)
    plt.plot(grid_points[end_coord, 0], grid_points[end_coord, 1], marker='x', linestyle='-', color='red', markersize=8)

    # Show the plot
    weights = grid_points[:, 2].reshape((y_length+1, x_length+1))
    plt.imshow(weights, origin='lower', extent=(0, x_length, 0, y_length), cmap='hot', interpolation='nearest')
    plt.colorbar(label='Cost')
    
    
    plt.title('Heatmap with Shortest Path')
    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')
    plt.savefig('heatmap.png')
    plt.show()
    
    
     

if __name__ == "__main__":
    main()
    