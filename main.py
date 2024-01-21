import matplotlib.pyplot as plt
import numpy as np
import matplotlib.collections as mcoll
import matplotlib.path as mpath
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import odeint
from matplotlib import animation

def hex_to_RGB(hex_str):
    """ #FFFFFF -> [255,255,255]"""
    #Pass 16 to the integer function for change of base
    return [int(hex_str[i:i+2], 16) for i in range(1,6,2)]

def get_color_gradient(c1, c2, n):
    """
    Given two hex colors, returns a color gradient
    with n colors.
    """
    assert n > 1
    c1_rgb = np.array(hex_to_RGB(c1))/255
    c2_rgb = np.array(hex_to_RGB(c2))/255
    mix_pcts = [x/(n-1) for x in range(n)]
    rgb_colors = [((1-mix)*c1_rgb + (mix*c2_rgb)) for mix in mix_pcts]
    return ["#" + "".join([format(int(round(val*255)), "02x") for val in item]) for item in rgb_colors]

initial_state = [0, 1, 0]

sigma = 10
rho = 28
beta = 8 / 3

# define the time points to solve for, evenly spaced between the start and end times
start_time = 0
# end_time = 500
end_time = 500
time_points = np.linspace(start_time, end_time, end_time * 100)

# define the lorenz system
def lorenz_system(current_state, t):
    x, y, z = current_state
    xdot = sigma * (y - x)
    ydot = x * (rho - z) - y
    zdot = x * y - beta * z
    return [xdot, ydot, zdot]  # returns the time derivatives

# use odeint() to solve a system of ordinary differential equations
# the arguments are:
# 1, a function - computes the derivatives
# 2, a vector of initial system conditions (aka x, y, z positions in space)
# 3, a sequence of time points to solve for
# returns an array of x, y, and z value arrays for each time point, with the initial values in the first row
xyz = odeint(lorenz_system, initial_state, time_points)

# extract the individual arrays of x, y, and z values from the array of arrays
x = xyz[:, 0]
y = xyz[:, 1]
z = xyz[:, 2]

# plot the lorenz attractor in three-dimensional phase space
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim((-30, 30))
ax.set_ylim((-30, 30))
ax.set_zlim((0, 50))
cols = get_color_gradient("#9f2c5c", "#00000", int(len(x) / 2)) + get_color_gradient("#000000", "#9f2c5c", len(x) - 1 - int(len(x) / 2))
# print(cols)
ax.set_facecolor("black")
plt.axis('off')
for i in range(len(x)-1):
    ax.plot(x[i:i+2], y[i:i+2], z[i:i+2], color=cols[i], alpha=0.7, linewidth=0.4)
# for i in range(0, len(x)-1, 4):
#     ax.plot(x[i:i+5], y[i:i+5], z[i:i+5], color=cols[i], alpha=0.7, linewidth=0.3)
ax.set_title('Lorenz attractor phase diagram')

def animate(i):
    print(i)
    ax.view_init(elev=10., azim=i)
    return fig,


xx, yy = np.meshgrid(range(-30, 31), range(-30, 31))
z = xx * 0 

# plot the plane
ax.plot_surface(xx, yy, z, alpha=0.1, linewidth=0, color='white')

# anim = animation.FuncAnimation(fig, animate,
#                                frames=360, interval=1, blit=True)
# anim.save('something.gif')
fig.savefig("something.png", dpi=1200, bbox_inches='tight')
plt.show()