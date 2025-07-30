from manim import *
import pandas as pd
import matplotlib.pyplot as plt

class Benchmark_Plot(MovingCameraScene):        
    def construct(self):
        # Load data
        df = pd.read_csv("benchmark_results.txt", delim_whitespace=True)

        # Unique scripts and sorted Ns
        scripts = sorted(df['script'].unique())
        Ns = sorted(df['N'].unique())

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