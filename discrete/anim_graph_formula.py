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

        # TODO: Compute the complexity
        
        # Discrete:
        # build_field + compute_heatmap + compute_graph + shortest_path
        # O(n)        + O(n)            + O(mn)         + O[n*k + n*log(n)] = O(nlogn)
        
        # Voronoi:
        # build_field + compute_voronoi + compute_weights + shortest_path
        # O(n)        + O(nlogn)        + O(mn)           + O[n*k + n*log(n)] = O(nlogn)
        
        # Voronoi_Union
        # build_field + compute_voronoi + compute_weights + sort_edges + union_find  + BFS
        # O(n)        + O(nlogn)        + O(mn)           + O(nlogn)   + O(alpha(n)) + O(kn) = O(nlogn)
        
        title = Text("Algorithm Complexity Comparison", font_size=40).move_to(UP*20)
        
        self.play(FadeIn(title))

        # --- Discrete ---
        # dicrete_text = Text(Discrete:)
        discrete_steps = VGroup(
            #MathTex(r"\textbf{Discrete:}"),
            MathTex(r"\underbrace{\mathcal{O}(n)}_{\text{build\_field}} + "),
            MathTex(r"\underbrace{\mathcal{O}(n)}_{\text{compute\_heatmap}} + "),
            MathTex(r"\underbrace{\mathcal{O}(mn)}_{\text{compute\_graph}} + "),
            MathTex(r"\underbrace{\mathcal{O}(nk + n \log n)}_{\text{shortest\_path}} = "),
            MathTex(r"\underbrace{\mathcal{O}(n \log n)}_{\text{Total}}")
        ).arrange(3*RIGHT, aligned_edge=RIGHT).scale(0.7).next_to(title, 2*DOWN, buff=0.3)

        # --- Voronoi ---
        voronoi_steps = VGroup(
            #MathTex(r"\textbf{Voronoi:}"),
            MathTex(r"\underbrace{\mathcal{O}(n)}_{\text{build\_field}} + "),
            MathTex(r"\underbrace{\mathcal{O}(n \log n)}_{\text{compute\_voronoi}} + "),
            MathTex(r"\underbrace{\mathcal{O}(mn)}_{\text{compute\_weights}} + "),
            MathTex(r"\underbrace{\mathcal{O}(nk + n \log n)}_{\text{shortest\_path}} = "),
            MathTex(r"\underbrace{\mathcal{O}(n \log n)}_{\text{Total}}")
        ).arrange(3*RIGHT, aligned_edge=RIGHT).scale(0.7).next_to(discrete_steps, 0.5*DOWN, buff=1.5)

        # --- Voronoi Union ---
        union_steps = VGroup(
            #MathTex(r"\textbf{Voronoi Union:}"),
            MathTex(r"\underbrace{\mathcal{O}(n)}_{\text{build\_field}} + "),
            MathTex(r"\underbrace{\mathcal{O}(n \log n)}_{\text{compute\_voronoi}} + "),
            MathTex(r"\underbrace{\mathcal{O}(mn)}_{\text{compute\_weights}} + "),
            MathTex(r"\underbrace{\mathcal{O}(n \log n)}_{\text{sort\_edges}} + "),
            MathTex(r"\underbrace{\mathcal{O}\alpha(n)}_{\text{union\_find}} + "),
            MathTex(r"\underbrace{\mathcal{O}(kn)}_{\text{BFS}} = "),
            MathTex(r"\underbrace{\mathcal{O}(n \log n)}_{\text{Total}}")
        ).arrange(3*RIGHT, aligned_edge=RIGHT).scale(0.7).next_to(voronoi_steps, 0.5*DOWN, buff=1.5)

        # Show final heading and equations
        self.play(self.camera.frame.animate.move_to((discrete_steps.get_center() + voronoi_steps.get_center())*0.5))
        self.wait()

        # Animate Discrete steps
        for step in discrete_steps:
            self.play(FadeIn(step), run_time=0.4)

        self.wait(0.5)

        # Animate Voronoi steps
        for step in voronoi_steps:
            self.play(FadeIn(step), run_time=0.4)

        self.wait(0.5)

        # Animate Voronoi Union steps
        for step in union_steps:
            self.play(FadeIn(step), run_time=0.3)

        self.wait(1)

        # Emphasize the "Total" rows
        for group in [discrete_steps, voronoi_steps, union_steps]:
            total_line = group[-1]
            self.play(total_line.animate.set_color(YELLOW).scale(1.1))

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