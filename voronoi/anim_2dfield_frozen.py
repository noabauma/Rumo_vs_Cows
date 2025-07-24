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
        
def is_inside_rect(point, rect: Rectangle):
    center = rect.get_center()
    half_w = rect.width / 2
    half_h = rect.height / 2

    x_min, x_max = center[0] - half_w, center[0] + half_w
    y_min, y_max = center[1] - half_h, center[1] + half_h

    return (
        x_min <= point[0] <= x_max and
        y_min <= point[1] <= y_max
    )

class TwoDField_Vor_Frozen(MovingCameraScene):        
    def construct(self):
        ##### Step 1: Defining the problem field
        x_length = 30        # x coordinate of the cows field [m]
        y_length = 20        # y coordinate of the cows field [m]
        
        obst_coord = np.array([[1/2, 3/4], [1/2, 1/4], [9/10, 3/4], [9/10, 1/4]])
        obst_coord = np.array([[1/2, 2/10], [1/2, 19/20], [1/10, 1/2], [9/10, 1/2]])
        obst_coord[:,0] *= x_length
        obst_coord[:,1] *= y_length
        
        n_obst = obst_coord.shape[0]
        
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
        
        # This Rect is for later to show the cost of one edge
        rect_domain = [x_length/3, y_length/3, [2/5*x_length, 2/5*y_length, 0]]
        rect_tmp = Rectangle(width=rect_domain[0], height=rect_domain[1]).move_to(rect_domain[2])
        
        
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
        
        vor_vertices_og  = VGroup()
        
        for x, y in vor.vertices:
            if (-1e-6 < x < x_length + 1e-6) and (-1e-6 < y < y_length + 1e-6):
                dot = Dot(point=[x, y, 0], radius=0.2, color=BLUE)
                vor_vertices_og.add(dot)

        
        vor_vertices_og.set_z_index(-2)

        # Animate
        # self.add(rect)
        self.add(cows)
        
        # Draw the starting and end point
        start_point = Circle(color=WHITE, fill_opacity=1).scale(0.6).move_to([vor.vertices[all_idx[0], 0], vor.vertices[all_idx[0], 1], 0])
        end_point = Star(color=WHITE, fill_opacity=1).scale(0.6).move_to([vor.vertices[all_idx[-1], 0], vor.vertices[all_idx[-1], 1], 0])

        # self.add(start_point)
        # self.add(end_point)
        
        # Now adding the edges (with labels)
        
        lines = VGroup()

        for ridge_vertex in vor.ridge_vertices: 
            i = ridge_vertex[0]
            j = ridge_vertex[1]
            
            #assert i != -1, f"something went wrong, we have a out-of-bounds ridge vertex: {vor.ridge_vertices}"
            
            if i == -1 or j == -1:
                continue
            
            a = vor.vertices[i]
            b = vor.vertices[j]
            
            if (-1e-6 < a[0] < x_length + 1e-6) and (-1e-6 < a[1] < y_length + 1e-6) and (-1e-6 < b[0] < x_length + 1e-6) and (-1e-6 < b[1] < y_length + 1e-6):
                line = Line(
                    [a[0], a[1], 0],
                    [b[0], b[1], 0],
                    stroke_width=4,
                    color=BLUE # interpolate_color(BLUE, RED, weight)
                )
                lines.add(line)
   
        lines.set_z_index(-1)

        self.add(vor_vertices_og)
        self.add(lines)
        
        self.wait()
        
        # # Draw the shortest path
        # path_lines = VGroup()
        # for i, j in zip(path[:-1], path[1:]):
        #     p1 = [vor.vertices[all_idx[i], 0], vor.vertices[all_idx[i], 1], 0]
        #     p2 = [vor.vertices[all_idx[j], 0], vor.vertices[all_idx[j], 1], 0]
        #     line = Line(p1, p2, color=PURE_GREEN, stroke_width=11)
        #     path_lines.add(line)
        
        # path_lines.set_z_index(2)

        # # Animate the path drawing
        # self.wait()
        # self.play(Create(path_lines), run_time=3)
        # self.wait()