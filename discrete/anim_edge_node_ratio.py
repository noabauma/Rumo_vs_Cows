from manim import *
import numpy as np

from main import * # all the functions of discrete/main.py

np.random.seed(42)   # seed for the random number generator


class Edge_node_ratio(MovingCameraScene):
    def construct(self):
        
        dot = Dot(point=[0, 0, 0], radius=0.2, color=BLUE)
        
        lines = VGroup()
        
        line = Line(
            [0, 0, 0],
            [-1, 0, 0],
            stroke_width=4,
            color=BLUE
        )
        lines.add(line)
        
        line = Line(
            [0, 0, 0],
            [-1, -1, 0],
            stroke_width=4,
            color=BLUE
        )
        lines.add(line)
        
        line = Line(
            [0, 0, 0],
            [0, -1, 0],
            stroke_width=4,
            color=BLUE
        )
        lines.add(line)
        
        line = Line(
            [0, 0, 0],
            [1, -1, 0],
            stroke_width=4,
            color=BLUE
        )
        lines.add(line)
        
        self.play(FadeIn(dot, lines))
        self.wait()
        
        group = VGroup(dot, lines)
        
        self.anim_u(group.copy())
        self.anim_r(group.copy())
    
            
    def anim_u(self, group: VGroup):
        self.play(group.animate.shift(UP), run_time=1)
        self.wait()
        
        self.rec_counter += 1
        if self.rec_counter < self.rec_max:
            self.anim_u(group.copy())
            self.anim_r(group.copy())
            
    def anim_r(self, group: VGroup):
        self.play(group.animate.shift(RIGHT), run_time=1)
        self.wait()
        
        self.rec_counter += 1
        if self.rec_counter < self.rec_max:
            self.anim_u(group.copy())
            self.anim_r(group.copy())
            
    rec_counter = 0
    rec_max = 16
        
    
