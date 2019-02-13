from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns
import numpy as np


def plot_score_hist(scores, tick_value=None):
    fig = Figure()
    canvas = FigureCanvas(fig)

    ax = fig.gca()
    ax = sns.distplot(scores, ax=ax)
    if tick_value is not None:
        ax.axvline(x=tick_value, color='r')

    canvas.draw()
    s, (width, height) = canvas.print_to_buffer()
    image = np.fromstring(s, np.uint8).reshape((height, width, 4))

    return image[:, :, :3]
