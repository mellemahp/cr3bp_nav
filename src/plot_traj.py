import json 
import sqlite3
from sql.queries import GetMsr
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from simulation.constants import R_E_ND, R_M_ND,  MU_EARTH_MOON
from simulation.stations import CR3BPEarthStn


def draw_sphere(ax, radius, center, color, mesh=False):
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
    y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]
    if mesh:
        ax.plot_wireframe(x, y, z, rstride=10, cstride=10, color=color)
    else: 
        ax.plot_surface(x, y, z, rstride=4, cstride=4, color=color)


def plot_moon(ax, scaling=10): 
    draw_sphere(ax, R_M_ND * scaling, [-1+MU_EARTH_MOON, 0, 0], 'gray')


def plot_earth(ax, scaling=10): 
    draw_sphere(ax, R_E_ND * scaling, [MU_EARTH_MOON, 0, 0], 'r', True)


def plot_stns(ax, conn, scaling=10):
     # extract stations from DB
    conn.row_factory = sqlite3.Row
    curs = conn.cursor()
    curs.execute('SELECT timestamp FROM truth_states ORDER BY timestamp DESC')
    times = [q[0] for q in curs.fetchall()]
    curs.execute('SELECT * from stations')
    station_list = [CR3BPEarthStn.from_db_object(dict(q)) for q in curs.fetchall()]

    for stn in station_list:
        pos_list = []
        for time in times: 
            pos_list.append(stn.state(time, include_shift=False)[:3] * scaling + np.array([MU_EARTH_MOON, 0.0, 0.0]))
        xs = [s[0] for s in pos_list]
        ys = [s[1] for s in pos_list]
        zs = [s[2] for s in pos_list]
        ax.plot3D(xs, ys, zs, label=stn.stn_name)


def plot_traj(ax, conn):
    c = conn.execute(GetMsr.ALL_TRUTH_STATES) 
    s = [ 
        json.loads(pos) + json.loads(vel) for pos, vel in  
        c.fetchall() 
    ]     
    xs = [x[0] for x in s]
    ys = [x[1] for x in s]
    zs = [x[2] for x in s]
    ax.plot3D(xs, ys, zs)


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().

    FROM: https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


if __name__ == "__main__": 
    scaling = 5
    conn = sqlite3.connect('./base_sim/sim1_msr.db') 

    fig = plt.figure(figsize = (10, 10))
    ax = fig.add_subplot(111, projection='3d')    
    plot_earth(ax, scaling*0.9)
    plot_moon(ax, scaling)
    plot_stns(ax, conn, scaling)
    plot_traj(ax, conn)
    set_axes_equal(ax)
    ax.legend()
    plt.show()