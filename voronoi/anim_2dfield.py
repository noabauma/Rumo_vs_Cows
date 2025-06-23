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
        
        vor_vertices_b   = VGroup()
        vor_vertices_og  = VGroup()
        vor_vertices_all = VGroup()
        
        for x, y in vor.vertices:
            if (1e-6 < x < x_length) and (1e-6 < y < y_length):
                dot = Dot(point=[x, y, 0], radius=0.2, color=BLUE)
                vor_vertices_og.add(dot)
                
            elif (-1e-6 < x < x_length + 1e-6) and (-1e-6 < y < y_length + 1e-6):
                dot = Dot(point=[x, y, 0], radius=0.2, color=ORANGE)
                vor_vertices_b.add(dot)
                
            dot = Dot(point=[x, y, 0], radius=0.2, color=BLUE)
            vor_vertices_all.add(dot)

        
        vor_vertices_og.set_z_index(0)
        vor_vertices_b.set_z_index(0)
        vor_vertices_all.set_z_index(0)

        # Animate
        self.play(Create(rect))
        self.play(LaggedStartMap(FadeIn, cows, lag_ratio=0.05))
        
        # Draw the starting and end point
        start_point = Circle(color=WHITE, fill_opacity=1).scale(0.6).move_to([vor.vertices[all_idx[0], 0], vor.vertices[all_idx[0], 1], 0])
        end_point = Star(color=WHITE, fill_opacity=1).scale(0.6).move_to([vor.vertices[all_idx[-1], 0], vor.vertices[all_idx[-1], 1], 0])

        self.play(Indicate(start_point, color=WHITE))
        self.wait()
        self.play(Indicate(end_point, color=WHITE))
        self.wait()
        
        # Now adding the edges (with labels)
        
        lines = VGroup()
        lines_b = VGroup()
        lines_all = VGroup()

        for i, j in vor.ridge_vertices:
            #assert i != -1, f"something went wrong, we have a out-of-bounds ridge vertex: {vor.ridge_vertices}"
            
            if i == -1 or j == -1:
                continue
            
            a = vor.vertices[i]
            b = vor.vertices[j]
            
            if (1e-6 < a[0] < x_length) and (1e-6 < a[1] < y_length) and (1e-6 < b[0] < x_length) and (1e-6 < b[1] < y_length):
                line = Line(
                    [a[0], a[1], 0],
                    [b[0], b[1], 0],
                    stroke_width=4,
                    color=BLUE # interpolate_color(BLUE, RED, weight)
                )
                lines.add(line)
            elif (-1e-6 < a[0] < x_length + 1e-6) and (-1e-6 < a[1] < y_length + 1e-6) and (-1e-6 < b[0] < x_length + 1e-6) and (-1e-6 < b[1] < y_length + 1e-6):
                line = Line(
                    [a[0], a[1], 0],
                    [b[0], b[1], 0],
                    stroke_width=4,
                    color=ORANGE # interpolate_color(BLUE, RED, weight)
                )
                lines_b.add(line)
                
            line = Line(
                [a[0], a[1], 0],
                [b[0], b[1], 0],
                stroke_width=4,
                color=BLUE # interpolate_color(BLUE, RED, weight)
            )
            lines_all.add(line)
                
        lines.set_z_index(-1)
        lines_b.set_z_index(-1)
        lines_all.set_z_index(-1)

        self.play(Create(vor_vertices_og))
        self.play(Create(lines), Create(lines_b))
        self.wait()
        
        # Draw the border points
        self.play(Indicate(vor_vertices_b))
        self.wait()
        
        self.play(
        self.camera.frame.animate.shift(0.5*y_length * UP).scale(2), run_time=2)
        self.play(FadeOut(vor_vertices_og, lines, vor_vertices_b, lines_b))
        self.wait()
        
        group_og = VGroup(rect, cows)#, vor_vertices_og, lines, lines_b)
        group_og_v = VGroup(vor_vertices_og, lines, lines_b)
        
        # Up
        mirror_u = group_og.copy()
        mirror_u.move_to(group_og.get_center())
        
        self.play(Rotate(mirror_u, angle=PI, axis=LEFT), run_time=2)
        self.play(mirror_u.animate.shift(y_length * UP), run_time=1)
        self.wait()
        
        self.play(self.camera.frame.animate.scale(0.5))
        
        mirror_u_v = group_og_v.copy()
        mirror_u_v.move_to(group_og_v.get_center())
        
        mirror_u_v.rotate(PI, axis=LEFT)
        mirror_u_v.shift(y_length * UP)
        
        self.play(FadeIn(mirror_u_v, vor_vertices_og, lines, lines_b))
        self.wait()
        self.play(Indicate(vor_vertices_b))
        self.wait()
        
        
        self.play(self.camera.frame.animate.move_to(rect), self.camera.frame.animate.scale(3), FadeOut(vor_vertices_b))
        self.camera.frame.save_state()
        
        # Left
        mirror_l = group_og.copy()
        mirror_l.move_to(group_og.get_center())
        
        self.play(Rotate(mirror_l, angle=PI, axis=UP), run_time=1)
        self.play(mirror_l.animate.shift(x_length * LEFT), run_time=0.5)
        
        # Up + Left
        mirror_ul = mirror_l.copy()
        mirror_ul.move_to(mirror_l.get_center())
        
        self.play(Rotate(mirror_ul, angle=PI, axis=RIGHT), run_time=1)
        self.play(mirror_ul.animate.shift(y_length * UP), run_time=0.5)
        
        # Right
        mirror_r = group_og.copy()
        mirror_r.move_to(group_og.get_center())
        
        self.play(Rotate(mirror_r, angle=PI, axis=DOWN), run_time=1)
        self.play(mirror_r.animate.shift(x_length * RIGHT), run_time=0.5)
        
        # Down + Right
        mirror_dr = mirror_r.copy()
        mirror_dr.move_to(mirror_r.get_center())
        
        self.play(Rotate(mirror_dr, angle=PI, axis=RIGHT), run_time=1)
        self.play(mirror_dr.animate.shift(y_length * DOWN), run_time=0.5)
        
        # Down + Left
        mirror_dl = mirror_l.copy()
        mirror_dl.move_to(mirror_l.get_center())
        
        self.play(Rotate(mirror_dl, angle=PI, axis=LEFT), run_time=1)
        self.play(mirror_dl.animate.shift(y_length * DOWN), run_time=0.5)
        
        # Up + Right
        mirror_ur = mirror_r.copy()
        mirror_ur.move_to(mirror_r.get_center())
        
        self.play(Rotate(mirror_ur, angle=PI, axis=LEFT), run_time=1)
        self.play(mirror_ur.animate.shift(y_length * UP), run_time=0.5)
        
        
        # Down
        mirror_d = group_og.copy()
        mirror_d.move_to(group_og.get_center())
        
        self.play(Rotate(mirror_d, angle=PI, axis=LEFT), run_time=1)
        self.play(mirror_d.animate.shift(y_length * DOWN), run_time=0.5)
        
        # Draw all voronoi nodes and edges
        self.play(FadeIn(vor_vertices_all, lines_all))
        self.wait()
        
        self.play(self.camera.frame.animate.scale(1./3.))
        self.wait()
        self.play(Indicate(vor_vertices_b))
        self.wait()
        
        
        """
        # Move out again to show the while field        
        margin = max(x_length, y_length)*0.1
        self.play(FadeOut(labels), 
                  FadeOut(line_labels), 
                  self.camera.frame.animate.move_to(rect).set(width=x_length + margin, height=y_length + margin))
        self.wait()
        
        # Save the state of camera
        self.camera.frame.save_state()
        
        
        
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