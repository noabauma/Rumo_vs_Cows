from manim import *
import numpy as np

from main import * # all the functions of discrete/main.py

np.random.seed(42)   # seed for the random number generator

class TreeDField_Vor(ThreeDScene):        
    def construct(self):
        ##### Step 1: Defining the problem field
        x_length = 30        # x coordinate of the cows field [m]
        y_length = 20        # y coordinate of the cows field [m]
        n_obst = 10          # number of obsticles (cows)
        
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
        # O[n*k + n*log(n)] with k in [3,6] -> O(n*log(n))
        dist_matrix, predecessors = shortest_path(csgraph=graph, method='auto', directed=False, indices=0, return_predecessors=True)
        
        # Backtrack to find the shortest path from source to destination
        path = []
        step = -1
        while step != 0:
            path.append(step)
            step = predecessors[step]

        path.append(0)
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
