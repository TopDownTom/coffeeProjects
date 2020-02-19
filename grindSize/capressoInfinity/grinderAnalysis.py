import time
import math
import numpy as np
import scipy
from scipy import optimize
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import csv
import pandas as pd
from definitions import *
import callFile
from scipy import stats as stats

def main(N,smallestIncrement):


    # attainable_mass_simulate and weighted_stddev are taken directly from grind size application source code, adapted only
    # to work with this code
    #Method to calculate attainable mass
    def attainable_mass_simulate(volumes):
            
            #This could be done better analytically
            depth_limit = 0.1 #mm
            
            radii = (3.0/4.0*volumes/np.pi)**(1/3)
            unreachable_volumes = np.full(volumes.size, 0.0)
            
            iboulders = np.where(radii > depth_limit)
            unreachable_volumes[iboulders[0]] = 4.0/3.0*np.pi*(radii[iboulders[0]] - depth_limit)**3
            reachable_volumes = volumes - unreachable_volumes
            
            return reachable_volumes

    def weighted_stddev(data, weights, frequency=True, unbiased=True):
            
            #Calculate the bias correction estimator
            if unbiased is True:
                    if frequency is True:
                            bias_estimator = (np.nansum(weights) - 1.0)/np.nansum(weights)
                    else:
                            bias_estimator = 1.0 - (np.nansum(weights**2))/(np.nansum(weights)**2)
            else:
                    bias_estimator = 1.0
            
            #Normalize weights
            weights /= np.nansum(weights)
            
            #Calculate weighted average
            wmean = np.nansum(data*weights)
            
            #Deviations from average
            deviations = data - wmean
            
            #Un-biased weighted variance
            wvar = np.nansum(deviations**2*weights)/bias_estimator
            
            #Un-biased weighted standard deviation
            wstddev = np.sqrt(wvar)
            
            return wstddev

    grindSetting=np.arange(1,N+1)
    ##### Extract Data from .csv files #####
    # Loop through each grind setting
    for a in range(1,N+1): # need to start at 1 because filenames are 1-N
        # import each file, list the values, then change the lists into arrays
        statsArray = np.array(pd.read_csv("data/setting%d_stats.csv" % a))
#        statsArray = np.array(statsList)

        
        # get the value from each stats column in the order:  '', avg_diam, std_diam, avg_surface, std_surface, efficiency, quality
        avgDiam.append(float(statsArray[:,1]))
        #stdDiam.append(float(statsArray[:,2]))
        avgSurf.append(float(statsArray[:,3])); stdSurf.append(float(statsArray[:,4]))
        efficiency.append(float(statsArray[:,5])); quality.append(float(statsArray[:,6]))

        # get the value from each .csv column in the order: ID, surface, roundness, short_axis, long_axis, volume, pixel_scale
        # data for grind setting 'a' can be retrieved as parameter[a]
        settingArray = np.array(pd.read_csv("data/setting%d.csv" % a))
#        settingArray = np.array(settingList)
        

        pixel_scale = settingArray[:,6]
        surfaces = settingArray[:,1]/pixel_scale**2
        volumes = settingArray[:,5]/pixel_scale**3
        attainable_masses = attainable_mass_simulate(volumes)
        data_weights = surfaces
        weights = np.maximum(np.ceil(attainable_masses/(coffee_cell_size/1e3)**3),1)
        surfacesAverage = np.sum(surfaces*weights)/np.sum(weights)
        stdDiamUpper.append(np.max(surfaces)-surfacesAverage)
        stdDiamLower.append(surfacesAverage-np.min(surfaces))
        surfacesStats = stats.describe(surfaces)
        skewness.append(surfacesStats[4])
        kurtosis.append(surfacesStats[5])

    # Calculate the average adjustment made between each whole-number grind setting and print results
    for b in range(0,N-1):
        settingAdjustment.append(avgSurf[b+1] - avgSurf[b])
    avgAdjustment = np.sum(settingAdjustment)/len(settingAdjustment)

    print()
    print("-----------------------------------------------")
    print("---------Grinder Adjustment Parameters --------")
    print()
    print("Total Adjustment Range (Setting {}-1): {:.2}mm^2".format(N,avgSurf[-1]-avgSurf[0]))
    print("Average Adjustment Between Each Setting: {:.2}mm^2".format(avgAdjustment))
    print()

    ##### Information To Plot #####                
    print()
    whichInformation = input("Which information would you like to view? (d)iameter,(s)urface: ")
    dataTypes=[avgDiam, stdDiam, avgSurf, stdSurf]
    def dataType(type):
        global data, dataError, pltTitle, units
        if type == "d":
            data = dataTypes[0]
            dataError = dataTypes[1]
            pltTitle = "Average Diameter"
            units = "mm"
        elif type == "s":
            data = dataTypes[2]
            dataError = dataTypes[3]
            pltTitle = "Average Surface Area"
            units = "mm^2"
        return data, dataError
    dataType(whichInformation)


    ##### Fitting #####
    print()
    # Ask user which regression form to use
    #fitType = input("Which Fit Type Would You Like? (l)inear,(q)uad: ")
    fitType = 'l'
    fitTypes = [funcLinear, funcQuad]
    def fittingFunction(type):
        global fitTypePlot

        # If the type is linear, perform all linear-regression related procedures
        if type == "l":
            popt, pcov = curve_fit(funcLinear, grindSetting, data, maxfev=2000) # the regression
            perr = np.sqrt(np.diag(pcov)) # error in the regression
            ss_res = np.sum((data - funcLinear(grindSetting,*popt))**2)
            ss_tot = np.sum ((data-np.mean(data))**2)
            r_squared = 1 - (ss_res/ss_tot)
            plt.plot(grindSetting, funcLinear(grindSetting, *popt), label="Linear Fit", color='green') # plots the regression against grind setting
            fitTypePlot = fitTypes[0]
#            plt.text(grindSetting[0],data[-1],r'$Equation\ of\ Linear\ Fit: y={:.2}x +({:.2})$'.format(popt[0],popt[1])) # generate equation of fit on figure
#            plt.text(grindSetting[0],data[9],r'$R^2={:.2}$'.format(r_squared)) # generate equation of fit r^2 value on figure
            print()
            print("------------- Fit Parameters ------------")       
            print("\n Slope = {:.2} +/- {:.2}".format(popt[0],perr[1]))
            print("\n Intercept = {:.2} +/- {:.2}mm".format(popt[1],perr[0]))
            print("\n R^2 = {:.2}".format(r_squared))
            print()

        elif type == "q":
            popt, pcov = curve_fit(funcQuad, grindSetting, data, maxfev=2000)
            perr = np.sqrt(np.diag(pcov))
            plt.plot(grindSetting, funcQuad(grindSetting, *popt), label="Quadratic Fit", color='green')
            plt.text(grindSetting[0],data[10],r'$Equation\ of\ Quadratic\ Fit: y={:.2}x^2+{:.2}x+{:.2}$'.format(popt[0],popt[1],popt[2]))
            fitTypePlot = fitTypes[1]

            print()
            print("------------- Fit Parameters ------------")       
            print("\n a: {:.2} +/- {:.2}".format(popt[0],perr[0]))
            print("\n b: {:.2} +/- {:.2}".format(popt[1],perr[1]))
            print("\n c: {:.2} +/- {:.2}".format(popt[2],perr[2]))
            print()
        return popt, pcov, fitTypePlot
    fittingFunction(fitType)


    ##### Plotting #####
#    input("Press Enter To Continue To Plots....")
    plt.title("{} vs. Grind Setting".format(pltTitle))
    plt.xlabel("Grind Setting")
    plt.xlim([0,N+1])
    plt.xticks(ticks=grindSetting)
    plt.ylabel("{} [{}]".format(pltTitle,units))

    plt.errorbar(grindSetting,data, fmt='o', color='black', ecolor='red', capsize=2, label='{}'.format(pltTitle), xerr=smallestIncrement/2, yerr=dataError)
#    plt.errorbar(grindSetting,data, fmt='o', color='black', ecolor='red', capsize=2, label='{}'.format(pltTitle), xerr=smallestIncrement/2, yerr=[stdDiamLower,stdDiamUpper])

    for i in range(0,len(avgDiam)):
        # Annotate the values for the errorbars on the graph, each for upper and lower.
        plt.annotate(data[i],(grindSetting[i]+.1*max(data),data[i]), color='black')
        plt.annotate(dataError[i],(grindSetting[i],dataError[i]+data[i]), color='red', label="error")
#        plt.annotate(data[i],(grindSetting[i],"{:.2}".format(skewness[i])), color='red', label="error")
#        plt.annotate("{:.2}".format(stdDiamLower[i]),(grindSetting[i],stdDiamLower[i]-data[i]),color='purple',label='lower error')
#        plt.annotate("{:.2}".format(stdDiamUpper[i]),(grindSetting[i],stdDiamUpper[i]+data[i]),color='orange',label='upper error')
    plt.legend()
    plt.savefig("{} Plot.png".format(pltTitle), dpi=199)
    plt.show()

