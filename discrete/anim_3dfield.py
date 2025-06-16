from manim import *
import numpy as np

from main import * # all the functions of discrete/main.py

np.random.seed(42)   # seed for the random number generator

class TwoDField(MovingCameraScene):        
    def construct(self):
        ##### Step 1: Let's build the problem field
        x_length = 30        # x coordinate of the cows field [m]
        y_length = 20        # y coordinate of the cows field [m]
        n_obst = 10          # number of obsticles (cows)
        
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

        ##### Step 5 manim the shit out of it!
        # TODO: Show it in the 3d view
        
        
