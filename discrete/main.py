import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

"""
This code computes the problem of Rumo having to pass a field with n cows.
We have to get from point A to point B with as little contact with the cows as possible, as he starts barking otherwise.
I.e. we are searching for the path of least resistance.
"""


def compute_heatmap(obst_coord: np.ndarray, x_length: float = 1, y_length: float = 1, grid_spacing: float = 1):
    """This function computes the heatmap of the cows as a grid of weighted nodes.
        The weight of a node is a heat map of how close rumo is allowed to come:
        cost = max(1 - dist/crit_dist, 0.1), i.e. 1 on top of a cow and 0.1 far away from all cows.

    Args:
        obst_coord (np.ndarray): The 2d coordinates of all the cows
        x_length (float, optional): The x length of the field. Defaults to 1.
        y_length (float, optional): The y length of the field. Defaults to 1.
        grid_spacing (float, optional): The spacing of the grid points [m]. Defaults to 1.

    Returns:
        np.ndarray: The grid points with their weights, shape (n_total_points, 3)
        int: The total number of grid points
        int: The number of grid points in the x-dimension
    """
    crit_dist = 10    # critical distance to a cow [m], getting closer than this will lead to rumo barking

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

    # compute the weights: the cost of a node only depends on its nearest cow,
    # so a single k-d tree query replaces a loop over every (node, cow) pair
    dist_to_nearest_cow, _ = cKDTree(obst_coord).query(grid_points[:, :2])
    grid_points[:, 2] = np.maximum(1 - dist_to_nearest_cow/crit_dist, 0.1)

    return grid_points, n_total_points, n_grid_points_x

def compute_graph(grid_points: np.ndarray, n_total_points: int, n_grid_points_x: int):
    """Turns the node weights of the grid into a weighted graph:
        every node is connected to its 8 neighbours and
        crossing an edge costs the average of its two node weights.

    Args:
        grid_points (np.ndarray): The grid points with their weights, shape (n_total_points, 3)
        n_total_points (int): The total number of grid points
        n_grid_points_x (int): The number of grid points in the x-dimension

    Returns:
        csr_matrix: The weighted graph
    """
    n_grid_points_y = n_total_points // n_grid_points_x
    node_costs = grid_points[:, 2]

    # x/y position of every node on the grid
    #
    #   y
    #   ^
    #   |
    #   |
    #   +----> x
    #
    node_idx = np.arange(n_total_points)
    node_x = node_idx % n_grid_points_x
    node_y = node_idx // n_grid_points_x

    # Build the graph directly in sparse (COO) format, one batch of edges per neighbour direction.
    # (A dense n_total_points x n_total_points matrix would need dozens of GB for big fields.)
    # We have to be careful with the boundaries as we don't want to go out of bounds.
    rows, cols, weights = [], [], []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue

            inside = (0 <= node_x + dx) & (node_x + dx < n_grid_points_x) & \
                     (0 <= node_y + dy) & (node_y + dy < n_grid_points_y)
            src = node_idx[inside]
            dst = src + dy*n_grid_points_x + dx

            rows.append(src)
            cols.append(dst)
            weights.append((node_costs[src] + node_costs[dst])/2)

    graph = csr_matrix((np.concatenate(weights), (np.concatenate(rows), np.concatenate(cols))),
                       shape=(n_total_points, n_total_points))

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
    if len(sys.argv) > 1:
        n_obst = int(sys.argv[1])
        seed = int(sys.argv[2])
        x_length = int((n_obst*100)**0.5)
        y_length = x_length

        np.random.seed(seed)
    else:
        x_length = 50        # x length of the cows field [m]
        y_length = 100       # y length of the cows field [m]
        n_obst = 100         # number of obstacles (cows)

        np.random.seed(43)   # seed for the random number generator

    grid_spacing = 1        # spacing of the grid points [m]

    obst_coord = np.random.rand(n_obst, 2)

    obst_coord[:,0] *= x_length
    obst_coord[:,1] *= y_length

    ##### Step 2: Computing the 2d heatmap of the cows in the field
    grid_points, n_total_points, n_grid_points_x = compute_heatmap(obst_coord, x_length, y_length, grid_spacing)

    ##### Step 3: Converting the heatmap into a weighted graph
    graph = compute_graph(grid_points, n_total_points, n_grid_points_x)

    # Define the starting and end points (as indices in the graph):
    # a random node on the lower boundary and a random node on the upper boundary
    start_coord = int(np.random.random_sample()*(n_grid_points_x - 1) + 0.5)
    end_coord = n_total_points - n_grid_points_x + int(np.random.random_sample()*(n_grid_points_x - 1) + 0.5)

    ##### Step 4: Compute the shortest path
    dist_matrix, predecessors = shortest_path(csgraph=graph, method='D', directed=False, indices=start_coord, return_predecessors=True)

    # Backtrack to find the shortest path from source to destination
    path = []
    step = end_coord
    while step != start_coord:
        path.append(step)
        step = predecessors[step]

    path.append(start_coord)
    path = path[::-1]  # Reverse the path to get it from source to destination

    print(time.time() - time_start, n_total_points)
    # print("total runtime: ", time.time() - time_start, "[s]")

    ##### Step 5: Plot

    plot = False
    if plot:
        # Plot the shortest path
        x_coords = grid_points[path, 0]
        y_coords = grid_points[path, 1]
        plt.plot(x_coords, y_coords, marker='o', linestyle='-', color='blue', markersize=8)

        # Plot the start and end points
        #plt.plot(grid_points[start_coord, 0], grid_points[start_coord, 1], marker='x', linestyle='-', color='green', markersize=8)
        #plt.plot(grid_points[end_coord, 0], grid_points[end_coord, 1], marker='x', linestyle='-', color='red', markersize=8)

        # Plot the heatmap
        n_grid_points_y = n_total_points // n_grid_points_x
        weights = grid_points[:, 2].reshape((n_grid_points_y, n_grid_points_x))
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
