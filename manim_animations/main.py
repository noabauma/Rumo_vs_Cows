from manim import *
import numpy as np

class RandomPointsInRectangle(Scene):
    def construct(self):
        x_length = 10        # x coordinate of the cows field [m]
        y_length = 5        # y coordinate of the cows field [m]
        n_obst = 100          # number of obsticles (cows)
        
        rect = Rectangle(width=x_length, height=y_length)
        self.add(rect)
        
        axes = Axes(
            x_range=[-x_length*0.5, x_length*0.5, 1],  # min, max, step for ticks
            y_range=[-y_length*0.5, y_length*0.5, 1],
            axis_config={"include_numbers": True, "color": WHITE},
        )
        self.add(axes)
        
        np.random.seed(42)   # seed for the random number generator
            
        grid_spacing = 1        # spacing of the grid points [m]
        
        obst_coord = np.random.rand(n_obst, 2)
        
        obst_coord[:,0] = obst_coord[:,0]*x_length - 0.5*x_length
        obst_coord[:,1] = obst_coord[:,1]*y_length - 0.5*y_length
        
        start_coord = int(np.random.random_sample()*x_length + 0.5)
        end_coord =  int(np.random.random_sample()*x_length + 0.5)  #+ grid_points.shape[0] - int(x_length/grid_spacing + 1)
        
        # Create Dot mobjects from the array
        points = VGroup(*[
            Dot(point=[x, y, 0], radius=0.05, color=YELLOW)
            for x, y in obst_coord
        ])

        # Animate
        self.play(Create(rect))
        self.play(LaggedStartMap(FadeIn, points, lag_ratio=0.05))
        self.wait()

