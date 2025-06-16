from manim import *
import numpy as np

class Graph_Formula(MovingCameraScene):        
    def construct(self):
        # Axes setup
        ax = Axes(
            x_range=[0, 20, 2],
            y_range=[0, 1, 0.1],
            x_length=9,
            y_length=3,
            axis_config={
                "color": WHITE,
                "include_ticks": True,
                "include_numbers": True,
                "decimal_number_config": {"num_decimal_places": 1},
            },
            x_axis_config={"numbers_to_include": [0, 10, 20]},
            y_axis_config={"numbers_to_include": [0, 0.1, 1.0]},
        )
        ax.to_edge(DOWN)  # Push axes downward to make room for formulas above

        # Dummy cost function
        def cost_function(dummy, x):
            crit = 10
            return max(1 - x / crit, 0.1)

        # Graph 1: piecewise
        graph1 = ax.plot(lambda x: cost_function([0], x), color=WHITE)

        # Graph 2: exponential
        graph2 = ax.plot(lambda x: np.exp(-x/10.0), color=BLUE)

        # Axis labels
        graph_labels = ax.get_axis_labels(x_label="\\text{distance } [m]", y_label="\\text{cost}")

        # Math formulas for graphs — centered above axes
        formula1 = MathTex(
            r"\text{cost}(r, c_i) = \max\left(1 - \frac{\|r - c_i\|_2}{\text{crit}},\ 0.1\right)"
        ).scale(0.6)

        formula2 = MathTex(
            r"\text{cost}(r, c_i) = e^{-\frac{\|r - c_i\|_2}{10}}", color=BLUE
        ).scale(0.6)

        formula_group = VGroup(formula1, formula2).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        formula_group.next_to(ax, UP, buff=1.0)  # Place above the graph

        # Add base elements
        self.add(ax, graph_labels)

        # Zoom
        self.play(
            self.camera.frame.animate.set(width=15).move_to(ax.get_center()),
            run_time=2
        )

        # Plot first graph and formula
        self.play(Create(graph1), Write(formula1))
        self.wait()

        # Plot second graph and second formula
        self.play(Create(graph2), Write(formula2))
        self.wait()

        # Final title and formulas (Cost summary) — further above
        title = Text("Cost Function for Rumo", font_size=36)
        title.move_to(UP*10)

        eq1 = MathTex(
            r"\text{cost}(r, c_i) = \max\left(1 - \frac{\|r - c_i\|_2}{\text{crit}},\ 0.1\right)"
        )
        eq2 = MathTex(
            r"\text{Cost}(r) = \max_{i \in \{1, \ldots, N\}} \text{cost}(r, c_i)"
        )

        eqs = VGroup(eq1, eq2).arrange(DOWN, buff=0.4).scale(0.9)
        eqs.next_to(title, DOWN, buff=0.3)

        # Show final heading and equations
        self.play(self.camera.frame.animate.move_to(eq1))
        self.wait()
        
        self.play(Write(title))
        self.wait(0.5)
        self.play(Write(eqs))
        self.wait(2)

        # TODO: Compute the complexity