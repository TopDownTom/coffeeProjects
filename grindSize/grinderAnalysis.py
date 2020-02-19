import time
import math
import numpy as np
import scipy
from scipy import optimize
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import csv
import pandas as pd

##### Definitions #####
# Define empty lists for each parameter & grind setting range
avgDiam=[]; stdDiam=[]
avgSurf=[]; stdSurf=[]
efficiency=[]; quality=[]
surface=[]
settingAdjustment=[]
# Ask how many grind settings were used during measurement
#N=np.int(input("Enter number of grind settings used: "))
N=11
grindSetting=np.arange(1,N+1)

# Define functions for use in error analysis
def funcLinear(xaxis,slope,intercept):
    return slope*xaxis+intercept

def funcQuad(xaxis,a,b,c):
    return a*xaxis**2+b*xaxis+c


##### Extract Data from .csv files #####
# Loop through each grind setting
for a in range(1,N+1): # need to start at 1 because filenames are 1-11
    # import each file, list the values, then change the lists into arrays
    statsList = pd.read_csv("ek43PreadjustmentData/setting%dc_stats.csv" % a )
    statsArray = np.array(statsList)

    
    # get the value from each stats column in the order:  '', avg_diam, std_diam, avg_surface, std_surface, efficiency, quality
    avgDiam.append(float(statsArray[:,1])); stdDiam.append(float(statsArray[:,2]))
    avgSurf.append(float(statsArray[:,3])); stdSurf.append(float(statsArray[:,4]))
    efficiency.append(float(statsArray[:,5])); quality.append(float(statsArray[:,6]))

    # get the value from each .csv column in the order ID, surface, roundness, short_axis, long_axis, volume, pixel_scale
    # data for grind setting 'a' can be retrieved as parameter[a]
    settingList = pd.read_csv("ek43PreadjustmentData/setting{}c.csv".format(a))
    settingArray = np.array(settingList)
    scale = settingArray[:,6].reshape(-1,1)
    surface.append(settingArray[:,1].reshape(-1,1)/scale**2)
    
    roundness = settingArray[:,2]/scale
    mass = settingArray[:,5]/scale**2
    volume = settingArray[:,5]/scale**3
#    attainable_masses = attainable_mass_simulate(volume)
    diameter = 2*np.sqrt(settingArray[:,4]*settingArray[:,3])/scale
#    diamWeight = np.max(np.ceil(attainable_masses/(coffee_cell_size/1e3)**3))
#    diamAverage = np.sum(diameter*diamWeight/np.sum(diamWeight))
#    data_weights = np.full(nclusters, 1)


# Calculate the average adjustment made between each whole-number grind setting and print results
for b in range(0,N-1):
    settingAdjustment.append(avgDiam[b+1] - avgDiam[b])
avgAdjustment = np.sum(settingAdjustment)/len(settingAdjustment)

#print("-----------------------------------------------")
#print("---------Grinder Adjustment Parameters --------")
#print()
#print("Total Adjustment Range (Setting {}-1): {:.2}mm".format(N,avgDiam[-1]-avgDiam[0]))
#print("Average Adjustment Between Each Setting: {:.2}mm".format(avgAdjustment))
#print()


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
fitType = input("Which Fit Type Would You Like? (l)inear,(q)uad: ")
fitTypes = [funcLinear, funcQuad]
def fittingFunction(type):
    global fitTypePlot

    # If the type is linear, perform all linear-regression related procedures
    if type == "l":
        popt, pcov = curve_fit(funcLinear, grindSetting, data, maxfev=2000) # the regression
        perr = np.sqrt(np.diag(pcov)) # error in the regression
        plt.plot(grindSetting, funcLinear(grindSetting, *popt), label="Linear Fit", color='green') # eventually plots the regression against grind setting
        fitTypePlot = fitTypes[0]
        plt.text(grindSetting[0],data[9],r'$Equation\ of\ Linear\ Fit: y={:.2}x+{:.2}$'.format(popt[0],popt[1])) # generate equation of fit on figure
        print()
        print("------------- Fit Parameters ------------")       
        print("\n Slope = {:.2} +/- {:.2}".format(popt[0],perr[1]))
        print("\n Intercept = {:.2} +/- {:.2}mm".format(popt[1],perr[0]))
        print()

    elif type == "q":
        popt, pcov = curve_fit(funcQuad, grindSetting, data, maxfev=2000)
        perr = np.sqrt(np.diag(pcov))
        plt.plot(grindSetting, funcQuad(grindSetting, *popt), label="Quadratic Fit", color='green')
        plt.text(grindSetting[0],data[9],r'$Equation\ of\ Quadratic\ Fit: y={:.2}x^2+{:.2}x+{:.2}$'.format(popt[0],popt[1],popt[2]))
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
input("Press Enter To Continue To Plots....")
plt.title("{} vs. Grind Setting".format(pltTitle))
plt.xlabel("Grind Setting")
plt.xlim([0,12])
plt.xticks(ticks=grindSetting)
plt.ylabel("{} [{}]".format(pltTitle,units))

plt.errorbar(grindSetting,data, fmt='o', color='black', ecolor='red', capsize=2, label='{}'.format(pltTitle), xerr=.05, yerr=dataError)

for i in range(0,len(avgDiam)):
    plt.annotate(data[i],(grindSetting[i]+.1*max(data),data[i]), color='black')
    plt.annotate(dataError[i],(grindSetting[i],dataError[i]+data[i]), color='red', label="error")

plt.legend()
plt.show()




'''
Scratch Stuff I Don't Want To Delete Yet
print("-----------------sk_learn----------------------")
fitXAxis = np.array(grindSetting).reshape(-1,1)
fitYAxis = np.array(avgDiam).reshape(-1,1)

model = LinearRegression().fit(fitXAxis,fitYAxis)

avgSizePrediction = model.predict(fitXAxis)

rSquared = model.score(fitYAxis,avgSizePrediction)

print()
print('Intercept:', model.intercept_)
print('Slope:', model.coef_)
print('R^2 Value:', rSquared)
plt.plot(grindSetting,avgSizePrediction, color='blue')
'''
