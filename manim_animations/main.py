from manim import *
import numpy as np

np.random.seed(42)   # seed for the random number generator

class RandomPointsInRectangle(MovingCameraScene):        
    def construct(self):
        x_length = 50        # x coordinate of the cows field [m]
        y_length = 50        # y coordinate of the cows field [m]
        n_obst = 10          # number of obsticles (cows)
        
       

        # Set camera frame size to fit the rectangle (with margin)
        margin = max(x_length, y_length)/10  # optional margin
        desired_width = x_length + margin
        desired_height = y_length + margin
        
        # Save the state of camera
        self.camera.frame.save_state()

        
        rect = Rectangle(width=x_length, height=y_length).move_to(x_length/2*RIGHT+y_length/2*UP)
        #self.add(rect)
        
        # Animation of the camera
        self.play(self.camera.frame.animate.set(width=desired_width, height=desired_height))
        self.wait()
        self.play(self.camera.frame.animate.move_to(rect))
        self.wait()
        
        # Save the state of camera
        self.camera.frame.save_state()
            
        grid_spacing = 1        # spacing of the grid points [m]
        
        obst_coord = np.random.rand(n_obst, 2)
        
        obst_coord[:,0] *= x_length
        obst_coord[:,1] *= y_length
        
        start_coord = int(np.random.random_sample()*x_length + 0.5)
        end_coord =  int(np.random.random_sample()*x_length + 0.5)  #+ grid_points.shape[0] - int(x_length/grid_spacing + 1)
        
        # Create Dot mobjects from the array
        points = VGroup(*[
            Dot(point=[x, y, 0], radius=0.5, color=LIGHT_BROWN)
            for x, y in obst_coord
        ])
        
        n_grid_points_x = int(x_length/grid_spacing + 1)    # number of grid points in the x-dimension
        n_grid_points_y = int(y_length/grid_spacing + 1)    # number of grid points in the y-dimension
        n_total_points = n_grid_points_x*n_grid_points_y
        
        # Step 1: Generate equally spaced points between in the x and y dimensions
        x_points = np.linspace(0, x_length, n_grid_points_x)
        y_points = np.linspace(0, y_length, n_grid_points_y)
        
        # Step 2: Create the 2D grid using meshgrid
        x, y = np.meshgrid(x_points, y_points)
        
        # Step 3: Combine the grid coordinates into an array of points
        grid_points = np.vstack([x.ravel(), y.ravel()]).T
        
        grid_points = VGroup(*[
            Dot(point=[x, y, 0], radius=0.1, color=RED)
            for x, y in grid_points
        ])

        # Animate
        self.play(Create(rect))
        self.play(LaggedStartMap(FadeIn, points, lag_ratio=0.05))
        self.play(Create(grid_points))
        self.wait()

