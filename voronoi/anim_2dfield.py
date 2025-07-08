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
    
def sigmoid_strengthen(x: float, alpha: float = 1.0):
    # alpha = 1/n with n in {1,3,5,7,...}
    return -0.5*(np.cos(x*np.pi)**alpha) + 0.5


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

        
        vor_vertices_og.set_z_index(-2)
        vor_vertices_b.set_z_index(-2)
        vor_vertices_all.set_z_index(-2)

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
        the_points = []

        for idx, ridge_point in enumerate(vor.ridge_points): 
            #assert i != -1, f"something went wrong, we have a out-of-bounds ridge vertex: {vor.ridge_vertices}"
            
            i = vor.ridge_vertices[idx][0]
            j = vor.ridge_vertices[idx][1]
            
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
                
            if is_inside_rect(a, rect_tmp) and is_inside_rect(b, rect_tmp):
                assert len(the_points) == 0, "This list should be empty! Only one line should exist inside this rect!"
                the_points = [a, b, obst_coord[ridge_point[0]], obst_coord[ridge_point[1]], cost_function(a, b, obst_coord[ridge_point[0]], obst_coord[ridge_point[1]])]
                
            line = Line(
                [a[0], a[1], 0],
                [b[0], b[1], 0],
                stroke_width=4,
                color=BLUE # interpolate_color(BLUE, RED, weight)
            )
            lines_all.add(line)
            
        assert len(the_points) > 0, "This list should not be empty! Only one line should exist inside this rect!"
                
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
        
        group_og = VGroup(rect, cows)
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
        
        # Filter based on y position
        visible_subset = VGroup(*[
            m for m in vor_vertices_b
            if abs(m.get_center()[1] - y_length) < 1e-6 and 0.0 < m.get_center()[0] < x_length
        ])
        visible_subset.set_color(BLUE)
        vor_vertices_b.set_color(BLUE)
        
        self.play(Indicate(visible_subset))
        self.wait()
        
        
        self.play(self.camera.frame.animate.move_to(rect).scale(3), FadeOut(visible_subset))
        
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
        
        lines.set_color(BLUE)
        lines_b.set_color(BLUE)
        
        self.play(FadeOut(vor_vertices_all, lines_all, mirror_u_v, mirror_u, mirror_d, mirror_l, mirror_r, mirror_dl, mirror_dr, mirror_ul, mirror_ur))
        self.wait()
        
        # TODO: Zoom in to one line and show cost of it
        # self.play(Create(rect_tmp))
        # self.wait()
        self.camera.frame.save_state()
        self.play(self.camera.frame.animate.move_to([2/5*x_length, 2/5*y_length, 0]).scale(3/10))
        self.wait()
        
        
        n = 10
        ts = np.linspace(0, 1, n)
        a  = the_points[0]
        b  = the_points[1]
        c1 = the_points[2]
        points = a[None, :] + ts[:, None] * (b - a)[None, :]  # shape (n, 2)
        
        # Compute cost at each sampled point (adjust your cost logic as needed)
        crit_dist = 10
        costs = np.maximum(1 - np.linalg.norm(points - c1, axis=1) / crit_dist, 0.1)
        
        discrete_points = VGroup()
        discrete_labels = VGroup()
        for i, point in enumerate(points):
            
            dot = Dot(point=[point[0], point[1], 0], radius=0.2, color=interpolate_color(BLUE, ORANGE, alpha=costs[i]))
            discrete_points.add(dot)
            
            label = DecimalNumber(
                        costs[i],
                        num_decimal_places=2,
                        font_size=20
                    ).move_to(dot.get_center() + 0.25*(UP+LEFT))
            discrete_labels.add(label)
            
        the_label =  DecimalNumber(
                        the_points[4],
                        num_decimal_places=2,
                        font_size=30
                    ).move_to(np.append((a+b)*0.5, 0) + 0.25*(UP+LEFT))   
        
        self.play(FadeIn(discrete_points))
        self.wait()
        
        self.play(Create(discrete_labels))
        self.wait()
        
        self.play(FadeOut(discrete_labels, discrete_points), Create(the_label))
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
        """
        
        self.play(FadeOut(the_label), self.camera.frame.animate.restore(), run_time=2)
        self.wait()
        
        # Draw the shortest path
        path_lines = VGroup()
        for i, j in zip(path[:-1], path[1:]):
            p1 = [vor.vertices[all_idx[i], 0], vor.vertices[all_idx[i], 1], 0]
            p2 = [vor.vertices[all_idx[j], 0], vor.vertices[all_idx[j], 1], 0]
            line = Line(p1, p2, color=PURE_GREEN, stroke_width=11)
            path_lines.add(line)
        
        path_lines.set_z_index(2)

        # Animate the path drawing
        self.play(Create(path_lines), run_time=3)
        self.wait()