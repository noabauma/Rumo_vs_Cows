from manim import *
import numpy as np

class Ackermann(MovingCameraScene):        
    def construct(self):
        
        title = Text("The Ackermann function")
        
        equation = MathTex(
            r"A(m, n) \equiv \begin{cases}"
            r"n + 1 & \text{if } m = 0\\"
            r"A(m-1, 1) & \text{if } n = 0\\"
            r"A(m-1,A(m,n-1)) & \text{if otherwise}"
            r"\end{cases}"
        )
        

        # Stack them vertically with spacing
        group = VGroup(title, equation).arrange(DOWN, buff=0.5)  # buff is spacing

        # Center the group on screen
        group.move_to(ORIGIN)

        # Animate
        self.play(Write(title))
        self.play(Write(equation), run_time=3)
        self.wait()
        
        self.play(FadeOut(group))
        self.wait()
        
        equation2 = MathTex(
            r"A(0,n) &= n+1 \overset{n=0}{\rightarrow} 1\\"
            r"A(1,n) &= n+2 \overset{n=1}{\rightarrow} 2\\"
            r"A(2,n) &= 2n+3 \overset{n=2}{\rightarrow} 7\\"
            r"A(3,n) &= 2^{n+3} - 3 \overset{n=3}{\rightarrow} 61\\"
            r"A(4,n) &= \underbrace{2^{2^{\dots^{2}}}}_{n+3} - 3 \overset{n=4}{\rightarrow} 2^{2^{2^{65536}}}"
        )
        
        self.play(Write(equation2), run_time=3)
        self.wait()
        
        self.play(FadeOut(equation2))
        self.wait()
        
        title2 = Text("The inverse Ackermann function")
        
        equation3 = MathTex(
            r"A(\alpha(n)) & = n\\"
            r"\alpha(n) & \gtrsim 3\\"
        )
        
        text = Text("(it will NEVER reach 4)", font_size=25)
        
        # Stack them vertically with spacing
        group = VGroup(title2, equation3, text).arrange(DOWN, buff=0.5)  # buff is spacing

        # Center the group on screen
        group.move_to(ORIGIN)
        
        # Animate
        self.play(Write(title2), runtime=3)
        self.play(Write(equation3), run_time=3)
        self.play(Write(text), runtime=3)
        self.wait()