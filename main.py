import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint


# This is used to convert a hex color to a RGB color
def hex_to_RGB(hex_str):
    """#FFFFFF -> [255,255,255]"""
    return [int(hex_str[i : i + 2], 16) for i in range(1, 6, 2)]


# This will give a color gradient from c1 to c2 (with n points in between) similar to linspace
def get_color_gradient(c1, c2, n):
    assert n > 1
    c1_rgb = np.array(hex_to_RGB(c1)) / 255
    c2_rgb = np.array(hex_to_RGB(c2)) / 255
    mix_pcts = [x / (n - 1) for x in range(n)]
    rgb_colors = [((1 - mix) * c1_rgb + (mix * c2_rgb)) for mix in mix_pcts]
    return [
        "#" + "".join([format(int(round(val * 255)), "02x") for val in item])
        for item in rgb_colors
    ]


# This will return the coordinates of the lorenz phase system
def get_lorenz_coords(initial_state, sigma, rho, beta, start_time, end_time):
    time_points = np.linspace(start_time, end_time, end_time * 100)

    def get_lorenz_time_derivatives(current_state, t):
        x, y, z = current_state
        xdot = sigma * (y - x)
        ydot = x * (rho - z) - y
        zdot = x * y - beta * z
        return [xdot, ydot, zdot]  # returns the time derivatives

    xyz = odeint(get_lorenz_time_derivatives, initial_state, time_points)
    return xyz


# This is to draw the given coordinates and save it to the given filename with the given dpi
def draw_and_save_lorenz_phase_diagram(xyz, filename, dpi):
    # Get the x, y, and z coordinates from the array
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim((-30, 30))
    ax.set_ylim((-30, 30))
    ax.set_zlim((0, 50))

    # This is to get a gradient like effect for the lines
    colors = get_color_gradient("#9f2c5c", "#00000", int(len(x) / 2)) + get_color_gradient("#000000", "#9f2c5c", len(x) - 1 - int(len(x) / 2))
    for i in range(len(x) - 1):
        ax.plot(x[i : i + 2], y[i : i + 2], z[i : i + 2], color=colors[i], alpha=0.7, linewidth=0.4)

    # plot the bottom xy-plane
    xx, yy = np.meshgrid(range(-30, 31), range(-30, 31))
    z = np.zeros(xx.shape)
    ax.plot_surface(xx, yy, z, alpha=0.1, linewidth=0, color="white")

    ax.set_title("Lorenz attractor phase diagram")
    ax.set_facecolor("black")
    plt.axis("off")

    fig.savefig(filename, dpi=dpi, bbox_inches="tight")


xyz = get_lorenz_coords(initial_state=[0, 1, 0], sigma=10, rho=28, beta=(8 / 3), start_time=0, end_time=500)
draw_and_save_lorenz_phase_diagram(xyz, "Lorenz Attractor Phase Diagram.png", 1200)
plt.show()