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
        
        group1 = group.copy()
        group2 = group.copy()
        group3 = group.copy()
        group4 = group.copy()
        
        self.play(group1.animate.shift(UP), group2.animate.shift(DOWN), group3.animate.shift(LEFT), group4.animate.shift(RIGHT))
        
        group1 = group1.copy()
        group2 = group2.copy()
        group3 = group3.copy()
        group4 = group4.copy()
        
        self.play(group1.animate.shift(LEFT), group2.animate.shift(RIGHT), group3.animate.shift(DOWN), group4.animate.shift(UP))
        
        group1 = group1.copy()
        group2 = group2.copy()
        group3 = group3.copy()
        group4 = group4.copy()
        
        self.play(group1.animate.shift(LEFT), group2.animate.shift(RIGHT), group3.animate.shift(DOWN), group4.animate.shift(UP))
        
        group1 = group1.copy()
        group2 = group2.copy()
        group3 = group3.copy()
        group4 = group4.copy()
        
        self.play(group1.animate.shift(DOWN), group2.animate.shift(UP), group3.animate.shift(RIGHT), group4.animate.shift(LEFT))
        
        group1 = group1.copy()
        group2 = group2.copy()
        group3 = group3.copy()
        group4 = group4.copy()
        
        self.play(group1.animate.shift(DOWN), group2.animate.shift(UP), group3.animate.shift(RIGHT), group4.animate.shift(LEFT))
        
        group1 = group1.copy()
        group2 = group2.copy()
        group3 = group3.copy()
        group4 = group4.copy()
        
        self.play(group1.animate.shift(DOWN), group2.animate.shift(UP), group3.animate.shift(RIGHT), group4.animate.shift(LEFT))
        
        self.play(Indicate(group))        
