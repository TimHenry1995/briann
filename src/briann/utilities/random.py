from typing import List, Iterator
from torch.utils.data import Sampler
import numpy as np

class StatelessRandom():
        
    def linear_congruential_generator(multiplier: int, modulo: int, seed: int, count: int) -> np.ndarray:
        
        # Compute array of random numbers
        result = np.zeros(shape=count)
        tmp = (seed*multiplier+1) % modulo
        result[0] = tmp/modulo
        for i in range(1, count):
            tmp = (tmp*multiplier+1) % modulo
            result[i] = tmp / modulo

        # Output
        return result

    def pseudo_uniform(lower_bound: float, upper_bound: float, seed: int, count: int) -> np.ndarray:

        # Get random numbers in range 0 to 1
        x = StatelessRandom.linear_congruential_generator(multiplier=3467541, modulo=(2**31)-1, seed=seed, count=count)

        # Scale
        x = lower_bound+(upper_bound-lower_bound)*x

        # Output
        return x

    def shuffle(indices: List[int], seed: int) -> List[int]:

        random_numbers = StatelessRandom.pseudo_uniform(lower_bound=0, upper_bound=len(indices), seed=seed, count=len(indices))

        # Iterate indices
        for i in range(len(indices)):
            # Find candidate to swop positions with
            j = (int)(random_numbers[i])
            
            # Swop
            tmp = indices[j]
            indices[j] = indices[i]
            indices[i] = tmp

class IndexSampler(Sampler[int]):

    def __init__(self, instance_count: int, seed: int = None) -> None:
        self._instance_count = instance_count
        self._seed = seed
        self._indices = list(range(instance_count))
        StatelessRandom.shuffle(indices=self._indices, seed=seed)

    @property
    def instance_count(self) -> int:
        return self._instance_count
    
    @property
    def seed(self) -> int:
        return self._seed

    def __len__(self) -> int:
        return self.instance_count
    
    def __iter__(self) -> Iterator[int]:
        for i in range(self.instance_count):
            yield self._indices[i]

if __name__ == "__main__":
    sampler = IndexSampler(instance_count=10, seed=5)
    for i in sampler:
        print(i)

    for i in sampler:
        print(i)

        
    import tkinter as tk
    import turtle

    def run_turtles(*args):
        for t, d in args:
            t.circle(200, d)
        root.after_idle(run_turtles, *args)

    def scroll_start(event):
        screen.scan_mark(event.x, event.y)

    def scroll_move(event):
        screen.scan_dragto(event.x, event.y, gain=1)

    root = tk.Tk()
    root.geometry("700x700")
    root.withdraw()

    frame = tk.Frame(bg='black')
    frame.pack(fill='both', expand=True)
    tk.Label(frame, text=u'Hello', bg='grey', fg='white').pack(fill='x')

    screen = turtle.ScrolledCanvas(frame)
    screen.pack(fill="both", expand=True)


    turtle1 = turtle.RawTurtle(screen)
    turtle2 = turtle.RawTurtle(screen)

    screen.bind("<ButtonPress-1>", scroll_start)
    screen.bind("<B1-Motion>", scroll_move)

    turtle1.ht(); turtle1.pu()
    turtle1.left(90); turtle1.fd(200); turtle1.lt(90)
    turtle1.st(); turtle1.pd()

    turtle2.ht(); turtle2.pu()
    turtle2.fd(200); turtle2.lt(90)
    turtle2.st(); turtle2.pd()

    root.deiconify()

    run_turtles((turtle1, 3), (turtle2, 4))

    root.mainloop()

        