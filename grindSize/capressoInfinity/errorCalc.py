import time
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
from scipy import stats as stats

import attainableMass
import definitions


def posNegError(data,whichInformation):

    ## ID, surface, roundness, shortaxis, longaxis, volume, pixelscale
    scale = data[:,6]
    coffee_cell_size=20
    default_binsize = .1
    default_log_binsize = .05
    default_bin_inflate = 1
    hist_color=[147, 36, 30]

    surfaces = data[:,1]/scale**2
    volumes = data[:,5]/scale**3
    attainable_masses = attainableMass.attainable_mass_simulate(volumes)
    weights = np.maximum(np.ceil(attainable_masses/(coffee_cell_size/1e3)**3),1)

    diameter = 2*np.sqrt(data[:,4]*data[:,3])/scale
    diamWeight = np.max(np.ceil(attainable_masses/(coffee_cell_size/1e3)**3))
    diamAverage = np.sum(diameter*diamWeight/np.sum(diamWeight))

    data_average = np.sum(surfaces*weights)/np.sum(weights)
    data_stddev = attainableMass.weighted_stddev(surfaces,weights,frequency=True,unbiased=True)

    ##### This is stuff for plotting a histogram #####
    if whichInformation == "d":
        data = diameter
        data_weights = diameter
    elif whichInformation == "s":
        data = surfaces
        data_weights = surfaces

    #Read x range from internal variables
    xmin = np.nanmin(data)
    xmax = np.nanmax(data)

    #Set histogram range
    histrange = np.array([xmin, xmax])

    #nbins = int(np.ceil( np.log10(float(histrange[1]) - np.log10(histrange[0]))/float(default_log_binsize*default_bin_inflate) ))

    nbins = int(np.ceil( float(histrange[1] - histrange[0])/float(default_binsize*default_bin_inflate) ))

    #Create a list of bins for plotting
    #    bins_input = np.logspace(np.log10(histrange[0]), np.log10(histrange[1]), nbins)
    bins_input = np.linspace(histrange[0], histrange[1], nbins)

    #Plot the histogram
    hist_color_fm = (hist_color[0]/255, hist_color[1]/255, hist_color[2]/255)
#    ypdf, xpdfleft, patches = plt.hist(data, bins_input, histtype='bar', label='hist_label', weights=data_weights/np.nansum(data_weights), density=False, lw=2, rwidth=.8)
    ypdf, xpdfleft = np.histogram(data, bins_input, range=None, weights=data_weights/np.nansum(data_weights), density=None) 

    #Find the value for the center of each bin
    xpdf = xpdfleft[0:-1] + np.diff(xpdfleft)/2.0

    #Calculate the average weighted by histogram height
    avg = np.nansum(ypdf*xpdf)/np.nansum(ypdf)

    #Create a cumulative density function (CDF) for the histogram
    ycdf = np.nancumsum(ypdf)/np.nansum(ypdf)

    #Find out positions of the CDF left and right of the average
    ileft = np.where(xpdf < avg)
    iright = np.where(xpdf >= avg)

    #Build an independently normalized CDF on the right side of the average
    ycdfpos = ycdf[iright[0]] - np.nanmin(ycdf[iright[0]])
    ycdfpos /= np.nanmax(ycdfpos)

    #Interpolate position that corresponds to 1-sigma positive error bar
    p1s = 0.68268949
    avg_plus_epos = np.interp(p1s,ycdfpos,xpdf[iright[0]])
    epos = avg_plus_epos - avg

    #Build an independently normalized CDF on the left side of the average
    ycdfneg = -ycdf[ileft[0]] - np.nanmin(-ycdf[ileft[0]])
    ycdfneg /= np.nanmax(ycdfneg)

    #Interpolate position that corresponds to 1-sigma negative error bar
    avg_min_eneg = np.interp(p1s, np.flip(ycdfneg, axis=0), np.flip(xpdf[ileft[0]], axis=0))
    eneg = avg - avg_min_eneg

    #Determine the vertical position where the "average" data point will be plotted
    ypos_errorbar = np.nanmax(ypdf)*0.05

    #Plot the "average" datapoint
    xerr = np.array([eneg, epos]).reshape(2, 1)
    marker = "o"
    markersize = 8
    elinewidth = 2
    capsize = 3
    capthick = 2
    #serr1 = plt.errorbar(avg, ypos_errorbar, xerr=xerr, marker=marker, markersize=markersize*1.4, linestyle="", color="w", elinewidth=elinewidth+2, capsize=capsize+1, capthick=capthick+1, alpha=0.8, zorder=19)
    #serr2 = plt.errorbar(avg, ypos_errorbar, xerr=xerr, marker=marker, markersize=markersize, linestyle="", color=hist_color_fm, elinewidth=elinewidth, ecolor='green', markeredgewidth=1.5, markeredgecolor="k", capsize=capsize, capthick=capthick, zorder=20)

    return epos,eneg
