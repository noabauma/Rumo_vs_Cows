from manim import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Benchmark_Plot(MovingCameraScene):        
    def construct(self):
        # Load the data
        df = pd.read_csv("benchmark_results.txt", delim_whitespace=True)

        # Get unique scripts and sorted N values
        scripts = sorted(df["script"].unique())
        colors = [BLUE, GREEN, RED]  # One color per script

        # Set N and log-scaled runtime
        min_N = df["N"].min()
        max_N = df["N"].max()
        max_runtime = df["avg_runtime"].max()

        # Manually calculate log10 range
        y_min = np.floor(np.log10(df["avg_runtime"].min()))
        y_max = np.ceil(np.log10(df["avg_runtime"].max()))

        # Axes (Note: y-range is in log10 space)
        axes = Axes(
            x_range=[min_N, max_N + 10, 50],
            y_range=[y_min, y_max, 1],  # log10 scale
            x_length=10,
            y_length=5,
            axis_config={"include_numbers": True},
            y_axis_config={
                "include_numbers": False,  # Disable default labels
                "label_direction": LEFT,
            },
        )
        axes.to_edge(LEFT)

        # Add log-scale y-axis labels (like 0.01, 0.1, 1, 10, 100)
        for exp in range(int(y_min), int(y_max) + 1):
            val = 10 ** exp
            y_pos = axes.c2p(min_N, exp)
            label = MathTex(f"{val:g}").scale(0.4).next_to(y_pos, LEFT, buff=0.15)
            self.add(label)
            
        # Axis labels
        x_label = Text("#cows", font_size=28).next_to(axes.x_axis.get_end(), RIGHT, buff=0.3)
        y_label = Text("avg time [s]", font_size=28).next_to(axes.y_axis.get_end(), UR, buff=0.3)

        # Add to scene
        self.play(Write(x_label), Write(y_label))

        self.play(Create(axes))

        # Plot one line per script
        graphs = []
        legends = VGroup()

        for script, color in zip(scripts, colors):
            data = df[df["script"] == script]
            N_vals = data["N"]
            runtime_vals = data["avg_runtime"]

            # Convert to log10 for plotting
            log_runtime_vals = np.log10(runtime_vals)

            # Create points
            points = [
                axes.c2p(N, log_runtime)
                for N, log_runtime in zip(N_vals, log_runtime_vals)
            ]

            # Line plot
            graph = VMobject(color=color).set_points_as_corners(points)  # straight lines
            graphs.append(graph)

            # Create and animate dots
            dot_group = VGroup(*[Dot(point=p, color=color, radius=0.05) for p in points])

            self.play(Create(graph), FadeIn(dot_group), run_time=1)
            
            # Add legend label
            label = Text(script.split("/")[0], font_size=24).set_color(color)
            legends.add(label)


        # Arrange legends
        legends.arrange(DOWN, aligned_edge=LEFT)
        legends.to_corner(UR)

        # # Animate drawing of graphs
        # for graph in graphs:
        #     self.play(Create(graph), run_time=1)

        self.play(FadeIn(legends))
        self.wait()

        """
        # Create matplotlib plot
        plt.figure(figsize=(6, 4))
        for script in scripts:
            sub_df = df[df['script'] == script]
            plt.plot(sub_df['N'], sub_df['avg_runtime'], marker='o', label=script)

        plt.xlabel("Number of Nodes (N)")
        plt.ylabel("Average Runtime (s)")
        plt.title("Benchmark Performance Comparison")
        plt.legend()
        plt.yscale('log')
        plt.grid(True)

        # Save the plot as an image
        plt.tight_layout()
        plt.savefig("benchmark_plot.png")
        plt.close()

        # Show it in Manim
        plot_image = ImageMobject("benchmark_plot.png").scale(1.2)
        self.play(FadeIn(plot_image))
        self.wait(3)
        """