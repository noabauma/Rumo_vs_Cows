from manim import *
import numpy as np

from main import * # all the functions of discrete/main.py

np.random.seed(42)   # seed for the random number generator

class Cross(VGroup):
    def __init__(self, point=ORIGIN, size=0.2, color=PURE_RED, stroke_width=4, **kwargs):
        line1 = Line((UP + LEFT) * size, (DOWN + RIGHT) * size, color=color, stroke_width=stroke_width, **kwargs)
        line2 = Line((DOWN + LEFT) * size, (UP + RIGHT) * size, color=color, stroke_width=stroke_width, **kwargs)
        super().__init__(line1, line2)
        self.move_to(point)

class TwoDField_Dis_Big(MovingCameraScene):        
    def construct(self):
        ##### Step 1: Let's build the problem field
        x_length = 100        # x coordinate of the cows field [m]
        y_length = 60        # y coordinate of the cows field [m]
        n_obst = 100          # number of obsticles (cows)
        
        np.random.seed(38)   # seed for the random number generator
            
        grid_spacing = 1        # spacing of the grid points [m]
        
        obst_coord = np.random.rand(n_obst, 2)
        
        obst_coord[:,0] *= x_length
        obst_coord[:,1] *= y_length

        ##### Step 2: Computing the 2d heatmap of the cows in the field
        grid_points, n_total_points, n_grid_points_x = compute_heatmap(obst_coord, x_length, y_length, grid_spacing)
        
        ##### Step 3: Converting the heatmap into a weighted graph
        graph = compute_graph(grid_points, n_total_points, n_grid_points_x)
        
        # Define the starting and end points (as indices in the graph)    
        start_coord = int(np.random.random_sample()*x_length + 0.5)  # 0
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
        
        rect.set_z_index(-2)
        
        # Move the camera to the rectangle
        margin = max(x_length, y_length)*0.1
        self.play(self.camera.frame.animate.move_to(rect).set(width=x_length + margin, height=y_length + margin))
        
        # Save the state of camera
        self.camera.frame.save_state()
        
        
        # Create Dot mobjects from the array
        cows = VGroup(*[
            Cross(point=[x, y, 0], size=0.2, color=PURE_RED, stroke_width=7)
            for x, y in obst_coord
        ])
        
        # Make the cows infront of the grid_points
        cows.set_z_index(1)
        
        grid_points_ = VGroup(*[
            Dot(point=[x, y, 0], radius=0.2, color=interpolate_color(BLUE, ORANGE, alpha=z))
            for x, y, z in grid_points
        ])
        
        grid_points_.set_z_index(0)

        # Animate
        self.add(rect)
        self.add(cows)
        
        # Draw the starting and end point
        start_point = Circle(color=WHITE, fill_opacity=1).scale(0.6).move_to([grid_points[start_coord, 0], grid_points[start_coord, 1], 0])
        end_point = Star(color=WHITE, fill_opacity=1).scale(0.6).move_to([grid_points[end_coord, 0], grid_points[end_coord, 1], 0])

        start_point.set_z_index(2)
        end_point.set_z_index(2)

        self.add(start_point)
        self.add(end_point)
        self.add(grid_points_)
        
        # Now adding the edges (with labels)
        lines = VGroup()
        coo = graph.tocoo()

        for i, j, weight in zip(coo.row, coo.col, coo.data):
            if i < j:
                start = grid_points[i]
                end = grid_points[j]

                line = Line(
                    [start[0], start[1], 0],
                    [end[0], end[1], 0],
                    stroke_width=2, # 1 + 3 * weight,
                    color=BLUE # interpolate_color(BLUE, RED, weight)
                )
                lines.add(line)
                
        lines.set_z_index(-1)

        self.add(lines)

        
        # Draw the shortest path
        path_lines = VGroup()
        for i, j in zip(path[:-1], path[1:]):
            p1 = [grid_points[i, 0], grid_points[i, 1], 0]
            p2 = [grid_points[j, 0], grid_points[j, 1], 0]
            line = Line(p1, p2, color=WHITE, stroke_width=10)
            path_lines.add(line)
        
        path_lines.set_z_index(2)

        # Animate the path drawing
        self.wait()
        self.play(Create(path_lines), run_time=3)
        self.wait()
