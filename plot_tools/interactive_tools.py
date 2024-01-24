import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from matplotlib.figure import Figure
from typing import Callable, TypeVar, List

T = TypeVar("T")

def example_plot_cycler():
    def plot(fig: Figure, a: int):
        formule = lambda x: a*x
        ax = fig.gca()
        ax.set_title("a =" + str(a))
        x = [0,1,2,3,4,5,6,7]
        y = [formule(xi) for xi in x]
        plot = ax.plot(x,y,'-',color='#000000')
        ax.set_ylim(0, 50)

    fig = plot_cycler(plot, [1,2,3,4,5,6,7,100])
    fig.show()

def plot_cycler(plot_fn: Callable[[Figure,T], None], inputs: List[T]) -> Figure:
    '''Create a figure that can cycle between plots with right and left keyboard keys'''

    class Counter:
        value = 0

    fig = plt.figure(figsize=(7, 9))
    current_idx = Counter()

    def update_plot(val):
        fig.clf()
        plot_fn(fig, val)
        plt.draw()

    def on_press(event):
        if (event.key == "left"):
            if current_idx.value <= 0:
                current_idx.value = len(inputs) -1
            current_idx.value -= 1
            update_plot(inputs[current_idx.value])
        if (event.key == "right"):
            if current_idx.value >= len(inputs) - 1:
                current_idx.value = 0
            current_idx.value += 1
            update_plot(inputs[current_idx.value])
        if (event.key == "up"):
            current_idx.value = len(inputs) - 1
            update_plot(inputs[current_idx.value])
        if (event.key == "down"):
            current_idx.value = 0
            update_plot(inputs[current_idx.value])

    fig.canvas.mpl_connect('key_press_event', on_press)
    update_plot(inputs[0])
    return fig