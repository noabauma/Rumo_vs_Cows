from manim import *
import numpy as np

from main import * # all the functions of discrete/main.py

np.random.seed(42)   # seed for the random number generator

class Discrete(MovingCameraScene):        
    def construct(self):
        time_start = time.time()
    
        ##### Step 1: Let's build the problem field
        x_length = 40        # x coordinate of the cows field [m]
        y_length = 25        # y coordinate of the cows field [m]
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
        self.play(self.camera.frame.animate.move_to(rect).set(width=x_length + margin, height=y_length + margin))
        
        # Save the state of camera
        self.camera.frame.save_state()

        # Create Dot mobjects from the array
        points = VGroup(*[
            Dot(point=[x, y, 0], radius=0.5, color=RED_A)
            for x, y in obst_coord
        ])
        
        # Make the cows infront of the grid_points
        points.set_z_index(1)
        
        grid_points_ = VGroup(*[
            Dot(point=[x, y, 0], radius=0.2, color=interpolate_color(BLUE, RED, alpha=z))
            for x, y, z in grid_points
        ])
        
        grid_points_.set_z_index(0)

        # Animate
        self.play(Create(rect))
        self.play(LaggedStartMap(FadeIn, points, lag_ratio=0.05))
        self.play(Create(grid_points_))
        self.wait()
        
        # Let's move the camera to a corner
        margin = max(x_length, y_length)/100  # optional margin
        camera_x = x_length/20
        camera_y = y_length*(19/20)
        view_width = (x_length/10 + margin)*(16/9)
        view_height = y_length/10 + margin
        self.play(self.camera.frame.animate.move_to([camera_x, camera_y, 0]).set(width=view_width, height=view_height))
        self.wait()
        
        # Save the state of camera
        self.camera.frame.save_state()
        
        # Animate the cost function here before adding the points
        ax = Axes(
            x_range=[0, 20, 2],       # [min, max, step] → tick marks every 1 unit
            y_range=[0, 1, 0.1],      # ticks every 0.2 on y-axis
            x_length=9,
            y_length=3,
            axis_config={
                "color": WHITE,
                "include_ticks": True,
                "include_numbers": True,
                "decimal_number_config": {"num_decimal_places": 1},
            },
            x_axis_config={
                "numbers_to_include": [0, 10, 20],  # customize labels if you want
            },
            y_axis_config={
                "numbers_to_include": [0, 0.1, 1.0],
            }
        )

        # Move to (-7, 4) for example
        ax.move_to([-10, 4, 0])

        # Plot your graph
        graph1 = ax.plot(lambda x: cost_function([0], x), color=WHITE)
        
        graph_labels = ax.get_axis_labels(x_label="distance [m]", y_label="cost")
        
        self.add(ax, graph1, graph_labels)
        
        # Zoom in slightly and move to the graph
        self.play(
            self.camera.frame.animate.set(width=12).move_to(ax.get_center()),
            run_time=2
        )
        self.wait()
        
        # Plot your graph
        graph2 = ax.plot(lambda x: np.exp(-x), color=BLUE)
        
        
        #self.add(graph2)
        self.play(Create(graph2), run_time=3)
        self.wait()
        
        
        self.play(Restore(self.camera.frame))
        self.wait()
        
        # Labels for z-values (heatmap values)
        labels = VGroup(*[
            DecimalNumber(z, num_decimal_places=2, color=WHITE)
            .scale(0.4)
            .next_to([x, y, 0], 2*UP, buff=0.1)
            for x, y, z in grid_points if abs(x - camera_x) <= view_width/2 and abs(y - camera_y) <= view_height/2
        ])
        
        labels.set_z_index(0)
        
        self.play(Create(labels))
        self.wait()
        
        # Now adding the edges (with labels)
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
                label = DecimalNumber(
                    weight,
                    num_decimal_places=2,
                    font_size=24
                ).move_to(line.get_center())

                # Optional: rotate label to match line direction
                angle = line.get_angle()
                label.rotate(angle)

                line_labels.add(label)
                
        lines.set_z_index(-1)
        line_labels.set_z_index(-1)

        self.play(LaggedStartMap(Create, lines, lag_ratio=0.01), run_time=5)
        self.wait()
        
        # TODO: Draw the labels of the lines
        
        # TODO: Draw the starting and end point
        
        # TODO: Draw the shortest path
        
        # TODO: Show it in the 3d view
        
        # TODO: Compute the complexity

