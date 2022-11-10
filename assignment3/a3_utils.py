import numpy as np
from matplotlib import pyplot as plt


def draw_line(rho, theta, h, w):
    """
    Example usage:

    plt.imshow(I)
    draw_line(rho1, theta1, h, w)
    draw_line(rho2, theta2, h, w)
    draw_line(rho3, theta3, h, w)
    plt.show()

    "rho" and "theta": Parameters for the line which will be drawn.
    "h", "w": Height and width of an image.
    """

    c = np.cos(theta)
    s = np.sin(theta)

    xs = []
    ys = []
    if s != 0:
        y = int(rho / s)
        if 0 <= y < h:
            xs.append(0)
            ys.append(y)

        y = int((rho - w * c) / s)
        if 0 <= y < h:
            xs.append(w - 1)
            ys.append(y)
    if c != 0:
        x = int(rho / c)
        if 0 <= x < w:
            xs.append(x)
            ys.append(0)

        x = int((rho - h * s) / c)
        if 0 <= x < w:
            xs.append(x)
            ys.append(h - 1)

    plt.plot(xs[:2], ys[:2], 'r', linewidth=.7)
