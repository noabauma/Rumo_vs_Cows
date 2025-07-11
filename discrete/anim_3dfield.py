from manim import *
import numpy as np

from main import * # all the functions of discrete/main.py

np.random.seed(42)   # seed for the random number generator

class TreeDField(ThreeDScene):        
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
        
        # At the border (i.e. Rectangle)
        rect = Rectangle(width=x_length, height=y_length).move_to(x_length/2*RIGHT+y_length/2*UP)
        
        # Move the camera to the rectangle
        margin = max(x_length, y_length)*0.1
        self.add(rect)
        self.set_camera_orientation(phi=60 * DEGREES, theta=30 * DEGREES)
        self.begin_ambient_camera_rotation(rate=0.1)
        self.wait(3)

        # Create Dot mobjects from the array
        cows = VGroup(*[
            Dot3D(point=[x, y, 1], radius=0.2, color=PURE_RED)
            for x, y in obst_coord
        ])
        
        # Make the cows infront of the grid_points
        cows.set_z_index(1)
        
        grid_points_ = VGroup(*[
            Dot3D(point=[x, y, z], radius=0.2, color=interpolate_color(BLUE, RED, alpha=z))
            for x, y, z in grid_points
        ])
        
        # Camera movement
        self.set_camera_orientation(phi=65 * DEGREES, theta=45 * DEGREES)
        self.begin_ambient_camera_rotation(rate=0.2)
        self.wait()
        self.stop_ambient_camera_rotation()
