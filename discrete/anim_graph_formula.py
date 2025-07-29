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
        ax.to_edge(3*DOWN)  # Push axes downward to make room for formulas above

        # Dummy cost function
        def cost_function(dummy, x):
            crit = 10
            return max(1 - x / crit, 0.1)

        # Graph 1: piecewise
        graph1 = ax.plot(lambda x: cost_function([0], x), color=WHITE)

        # Graph 2: exponential
        graph2 = ax.plot(lambda x: np.exp(-x/10.0), color=BLUE)

        # Axis labels
        #graph_labels = ax.get_axis_labels(x_label="\\text{distance } [m]", y_label="\\text{cost}")

        # Create custom axis labels
        x_label = MathTex("\\text{distance } [m]").next_to(ax.c2p(5, 0), DOWN)
        y_label = MathTex("\\text{cost}").next_to(ax.c2p(0, 0.5), LEFT).rotate(0.5*PI)

        self.add(x_label, y_label)

        # Math formulas for graphs — centered above axes
        formula1 = MathTex(
            r"\text{cost}(r, c_i) = \max\left(1 - \frac{\|r - c_i\|_2}{\text{crit}},\ 0.1\right)"
        ).scale(1.0)

        formula2 = MathTex(
            r"\text{cost}(r, c_i) = e^{-\frac{\|r - c_i\|_2}{10}}", color=BLUE
        ).scale(1.0)

        formula_group = VGroup(formula1, formula2).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        formula_group.next_to(ax, 0.5*UP, buff=1.0)  # Place above the graph

        # Add base elements
        self.add(ax)

        # Zoom
        self.play(
            self.camera.frame.animate.set(width=15).move_to(ax.get_center()+UP),
            run_time=2
        )

        # Plot first graph and formula
        self.play(Create(graph1), Write(formula1))
        self.wait()

        # Plot second graph and second formula
        self.play(Create(graph2), Write(formula2))
        self.wait()
        
        self.play(FadeOut(graph2, formula2))
        self.wait()

        # Final title and formulas (Cost summary) — further above
        title = Text("Cost Function of Rumo", font_size=36)
        title.move_to(UP*10)

        eq1 = MathTex(
            r"\text{cost}(r, c_i) = \max\left(1 - \frac{\|r - c_i\|_2}{\text{crit}},\ 0.1\right)"
        )
        eq2 = MathTex(
            r"\text{Cost}(r) = \max_{i \in \{1, \ldots, n\}} \text{cost}(r, c_i)"
        )

        eqs = VGroup(eq1, eq2).arrange(2*DOWN, buff=0.4).scale(0.9)
        eqs.next_to(title, 2*DOWN, buff=0.3)

        # Show final heading and equations
        self.play(self.camera.frame.animate.move_to(eq1))
        self.wait()
        
        self.play(Write(title))
        self.wait(0.5)
        self.play(Write(eqs))
        self.wait(2)

        # Compute the complexity
        
        # How the edges scale by number of vor.vertices
        # The number of vertices in a voronoi diagram of a set of n points is at most 2n-5 and the number of edges are at most 3n-6.
        # https://www.cse.iitk.ac.in/users/amit/courses/RMP/presentations/kshitij/index.html#:~:text=The%20number%20of%20vertices%20in,the%20convex%20hull%20of%20P.
        # C: 0, 1, 2, 3,
        # V: 0, 0, 0, 1, 2, 3, 4, 5
        # E: 0, 0, 1, 3, 5, 7
        
        title = Text("Algorithm Complexity Comparison", font_size=40).move_to(UP*20)
        
        self.play(FadeIn(title))
        
        # --- Discrete ---
        discrete_label = Text("Discrete:", font_size=20).next_to(title, 1*DOWN, buff=0.3)
        discrete_expr = MathTex(
            #r"\underbrace{\mathcal{O}(c + n)}_{\text{build field}} + "
            r"\underbrace{\mathcal{O}(cn)}_{\text{compute heatmap}} + "
            r"\underbrace{\mathcal{O}(kn)}_{\text{compute graph}} + "
            r"\underbrace{\mathcal{O}((kn + n) \log n)}_{\text{shortest path}} \overset{c<<n,k=4}{=}"
        )
        discrete_total = MathTex(
            r"\underbrace{\mathcal{O}(n \log n)}_{\text{Total}}"
        )
        discrete_group = VGroup(discrete_expr, discrete_total).scale(0.7).arrange(RIGHT, buff=0.15)
        discrete_group.next_to(discrete_label, DOWN, buff=0.3)  # <- FIXED

        # --- Voronoi ---
        voronoi_label = Text("Voronoi:", font_size=20).next_to(discrete_label, 1.5*DOWN, buff=1.3)  # align by label
        voronoi_expr = MathTex(
            #r"\underbrace{\mathcal{O}(c)}_{\text{build field}} + "
            r"\underbrace{\mathcal{O}(c \log (kn))}_{\text{compute voronoi}} + "
            r"\underbrace{\mathcal{O}(kn)}_{\text{compute weights}} + "
            r"\underbrace{\mathcal{O}((kn + n) \log n)}_{\text{shortest path}} \overset{c\approx\frac{n}{2},k\approx3}{=}"
        )
        voronoi_total = MathTex(
            r"\underbrace{\mathcal{O}(n \log n)}_{\text{Total}}"
        )
        voronoi_group = VGroup(voronoi_expr, voronoi_total).scale(0.7).arrange(RIGHT, buff=0.15)
        voronoi_group.next_to(voronoi_label, DOWN, buff=0.3)  # <- FIXED

        # --- Voronoi Union ---
        union_label = Text("Voronoi Union Find:", font_size=20).next_to(voronoi_label, 1.5*DOWN, buff=1.3)  # align by label
        union_expr = MathTex(
            #r"\underbrace{\mathcal{O}(c)}_{\text{build field}} + "
            r"\underbrace{\mathcal{O}(c \log (kn))}_{\text{compute voronoi}} + "
            r"\underbrace{\mathcal{O}(kn)}_{\text{compute weights}} + "
            r"\underbrace{\mathcal{O}(kn \log (kn))}_{\text{sort edges}} + "
            r"\underbrace{\mathcal{O}(n \alpha(n))}_{\text{union find}} + "
            r"\underbrace{\mathcal{O}(kn + n)}_{\text{DFS}} \overset{c\approx\frac{n}{2},k\approx3}{=}"
        )
        union_total = MathTex(
            r"\underbrace{\mathcal{O}(n \log n)}_{\text{Total}}"
        )
        union_group = VGroup(union_expr, union_total).scale(0.7).arrange(RIGHT, buff=0.15)
        union_group.next_to(union_label, DOWN, buff=0.3)



        # Show final heading and equations
        self.play(self.camera.frame.animate.move_to(voronoi_label))
        self.wait()
        
        all_labels = VGroup(discrete_label, voronoi_label, union_label)
        self.play(FadeIn(all_labels))

        # Animate Discrete steps
        self.play(FadeIn(discrete_expr))

        self.wait(0.5)

        # Animate Voronoi steps
        self.play(FadeIn(voronoi_expr))

        self.wait(0.5)

        # Animate Voronoi Union steps
        self.play(FadeIn(union_expr))

        self.wait(1)

        # Emphasize the "Total" rows
        for total in [discrete_total, voronoi_total, union_total]:
            self.play(total.animate.set_color(YELLOW).scale(1.1))

        self.wait(2)
        
        # TODO: Quote the thing what Raphael said
        quote1 = Paragraph(
                    "Find a path from A to B such that the\n"
                    "sum of all edge weights is minimized.",
                    alignment="center", font_size=35, t2w={'sum of all edge weights is minimized':BOLD}).move_to(UP*30)
        quote2 = Paragraph("Find a path from A to B such that the\n"
                      "maximum edge weight along the path is minimized.", 
                      alignment="center", font_size=35, t2w={'maximum edge weight along the path is minimized':BOLD}).move_to(UP*28)
        
        self.play(self.camera.frame.animate.move_to((quote1.get_center() + quote2.get_center())*0.5))
        self.wait()
        
        self.play(FadeIn(quote1))
        self.wait()
        self.play(FadeIn(quote2))
        self.wait()
        
        # TODO: Add a 3d visualization of the cost function