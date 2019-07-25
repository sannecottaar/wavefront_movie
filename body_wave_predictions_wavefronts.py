# -*- coding: utf-8 -*-
#############################################################################
### COMPUTES RAY PATHS AND TRAVEL TIMES FOR DIFFERENT BODY PHASES ##########
#############################################################################

### Importing various python libraries
# numpy is a useful toolkit for scientific computations
import numpy as np
# matplotlib is a plotting toolkit
import matplotlib.pyplot as plt
# Obspy is a seismic toolkit
import obspy
from obspy.taup import TauPyModel
from obspy.taup import plot_travel_times
from obspy.taup import plot_ray_paths

import matplotlib
from matplotlib.animation import FuncAnimation

# More about the obspy routines we are using can be found here:
# https://docs.obspy.org/packages/obspy.taup.html

# velocity model as a function of depth.
model = TauPyModel(model='ak135')

#######################################################
### You don't have to change anything but the four lines
### below to do the practical. Feel free to experiment
### though, but maybe save a copy for reference first.

# Phases to plot, e.g. plotphase =["PKJKP", "SKKS"]
# See phase naming information below or on the link above.
phases_to_plot =['P','Pdiff','PKP','PKIKP', 'PcP', 'PKKP', 'PP']
# min and max distances for the different phases to consider
# these can be chosen broadly (could be set 0-180 for all), it just slows down the code a bit.
rays_dist_min=[0, 100,100, 130, 0, 60, 0]
rays_dist_max=[110, 150,  181, 181, 160,180, 180 ]

# darkness of grey (lower is lighter here)
# this should be adapted, colors should change at reflection points
color_attenuation=[1.0, 0.5, 0.8, 0.8, 0.3, 0.8, 1.0]

# depth of earthquake in km
depth_earthquake= 30.
# radius of Earth in km
radius = 6371.

# regular time array in s, 1 s resolution
time = np.arange(0., 2000.)


# calculate through and save ray paths in interpolated time domain
save_paths=[]
for p, phase  in enumerate(phases_to_plot):
    dists_collected=[]
    depths_collected=[]
    for r, dist in enumerate(np.arange(rays_dist_min[p], rays_dist_max[p], 1)): # resolution could be improved here
        # get raypaths
        rays = model.get_ray_paths(depth_earthquake, dist, phase_list=[phase])
        # Loop through rays found, some phases have multiple paths
        for ray in rays:
            # Interpolate to regulard time array
            dists = np.interp(time, ray.path['time'], ray.path['dist'], left = np.nan, right = np.nan)
            depths = np.interp(time, ray.path['time'], ray.path['depth'], left = np.nan, right = np.nan)
            # save paths
            dists_collected.append(dists)
            depths_collected.append(depths)
    save_paths.append([np.array(dists_collected),np.array(depths_collected)])

save_paths=np.array(save_paths)

#interpolater for plotter
intp = matplotlib.cbook.simple_linear_interpolation

# save wave fronts to the left and right
lines_left=[]
lines_right=[]

# start plot for t= 0
fig = plt.figure(figsize =(10,5))
# define polar subplot
ax = plt.subplot(1,1,1, polar = True)
ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)
ax.set_xticks([])
ax.set_yticks([])

# plot paths for t=0 (these are all at the earthquake
t=0
for p in range(len(save_paths)):
    dists_collected=save_paths[p,0]
    depths_collected=save_paths[p,1]
    front_dist= dists_collected[:,t] # Cut at single time across all paths
    front_depth = depths_collected[:, t] # Cut at single time across all paths
    # set colour
    col = ((1-(1.-float(0)/2000 )*color_attenuation[p])+0.5)/1.5
    cols = [col, col,col]
    # plot line towards the right side
    line, = ax.plot(intp(front_dist, 100),radius - intp(front_depth, 100),color =cols )
    lines_right.append(line)
    # mirror line towards the left
    line, = ax.plot(intp(-1.*front_dist, 100),radius - intp(front_depth, 100),color =cols)
    lines_left.append(line)

# add discontinuities
discons = rays.model.s_mod.v_mod.get_discontinuity_depths()
ax.set_yticks(radius - discons)
ax.xaxis.set_major_formatter(plt.NullFormatter())
ax.yaxis.set_major_formatter(plt.NullFormatter())

# Pretty earthquake marker.
ax.plot([0], [radius - depth_earthquake],
        marker="*", color="#FEF215", markersize=20, zorder=10,
        markeredgewidth=1.5, markeredgecolor="0.3",
        clip_on=False)

ax.set_rmax(radius)
ax.set_rmin(0.0)
#plt.show()

def animate(t, lines_left, lines_right):
    '''
        Function updates lines with time
    '''
    global save_paths

    for l,line in enumerate(lines_right):
        dists_collected=save_paths[l,0]
        depths_collected=save_paths[l,1]
        front_dist= dists_collected[:,t]
        front_depth = depths_collected[:, t]
        # update line to the right
        line.set_xdata(intp(front_dist,100))
        line.set_ydata(radius - intp(front_depth,100))
        # mirror update for lines to left
        lines_left[l].set_xdata(intp(-1.*front_dist,100))
        lines_left[l].set_ydata(radius - intp(front_depth,100))
        # update colour
        col = ((1-(1.-float(t)/2000 )*color_attenuation[l])+0.5)/1.5
        cols = [col, col,col]
        line.set_color(cols)
        lines_left[l].set_color(cols)
    return(line,)

# Sets up animation
animation = FuncAnimation(
                          # Your Matplotlib Figure object
                          fig,
                          # The function that does the updating of the Figure
                          animate,
                          # Frame information (here just frame number)
                          np.arange(1000),
                          # Extra arguments to the animate function
                          fargs=[lines_left, lines_right],
                          # Frame-time in ms; i.e. for a given frame-rate x, 1000/x
                          interval=1000 / 25
                          )
animation.save("out_funcanimation.mp4")

