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

class TwoDField_Vor(MovingCameraScene):        
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
        self.play(self.camera.frame.animate.move_to(rect).set(width=x_length + margin, height=y_length + margin))
        
        # Save the state of camera
        self.camera.frame.save_state()
        
        # Create Dot mobjects from the array
        cows = VGroup(*[
            Cross(point=[x, y, 0], size=0.2, color=PURE_RED, stroke_width=7)
            for x, y in og_obst_coord
        ])
        
        # Make the cows infront of the grid_points
        cows.set_z_index(1)
        
        og_obst_coord_ = VGroup(*[
            Dot(point=[vertex[0], vertex[1], 0], radius=0.2, color=BLUE)
            for vertex in vor.vertices
        ])
        
        og_obst_coord_.set_z_index(0)

        # Animate
        self.play(Create(rect))
        self.play(LaggedStartMap(FadeIn, cows, lag_ratio=0.05))
        self.play(Create(og_obst_coord_))
        self.wait()
        
        
        # Now adding the edges (with labels)
        """
        lines = VGroup()
        line_labels = VGroup()
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
                
                # Label
                x = line.get_center()[0]
                y = line.get_center()[1]
                if abs(x - camera_x) < view_width/2 and abs(y - camera_y) < view_height/2:
                
                    label = DecimalNumber(
                        weight,
                        num_decimal_places=2,
                        font_size=20
                    ).move_to(line.get_center())
                    
                    # Rotate label to match line direction
                    angle = line.get_angle()
                    label.rotate(angle)
                    
                    if 0.5 < angle < 1.0:
                        label.move_to(label.get_center() + 0.20*(UP + RIGHT))
                    elif 2.0 < angle < 2.5:
                        label.move_to(label.get_center() + 0.20*(UP + LEFT))
                        label.rotate(PI)
                    else:
                        label.move_to(label.get_center() + 0.1*(UP))
                    
                    line_labels.add(label)
                
        lines.set_z_index(-1)
        line_labels.set_z_index(-1)

        self.play(LaggedStartMap(Create, lines, lag_ratio=0.01), run_time=5)
        self.wait()
        
        # Draw the labels of the lines
        self.add(line_labels)
        self.wait()
        
        # Move out again to show the while field        
        margin = max(x_length, y_length)*0.1
        self.play(FadeOut(labels), 
                  FadeOut(line_labels), 
                  self.camera.frame.animate.move_to(rect).set(width=x_length + margin, height=y_length + margin))
        self.wait()
        
        # Save the state of camera
        self.camera.frame.save_state()
        
        # TODO: Draw the starting and end point
        start_point = Circle(color=WHITE, fill_opacity=1).scale(0.6).move_to([grid_points[start_coord, 0], grid_points[start_coord, 1], 0])
        end_point = Star(color=WHITE, fill_opacity=1).scale(0.6).move_to([grid_points[end_coord, 0], grid_points[end_coord, 1], 0])

        self.play(Indicate(start_point, color=WHITE))
        self.wait()
        self.play(Indicate(end_point, color=WHITE))
        self.wait()
        
        # TODO: Draw the shortest path
        path_lines = VGroup()
        for i, j in zip(path[:-1], path[1:]):
            p1 = [grid_points[i, 0], grid_points[i, 1], 0]
            p2 = [grid_points[j, 0], grid_points[j, 1], 0]
            line = Line(p1, p2, color=WHITE, stroke_width=10)
            path_lines.add(line)
        
        path_lines.set_z_index(2)

        # Animate the path drawing
        self.play(Create(path_lines), run_time=3)
        self.wait()
        """