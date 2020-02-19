import time
import math
import numpy as np
import scipy
from scipy import optimize
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import csv

##### Definitions #####
# Define empty lists for each parameter & grind setting range
avgDiameter=[]
stdDiameter=[]
avgSurface=[]
stdSurface=[]
efficiency=[]
quality=[]
settingAdjustment=[]
grindSetting=np.arange(1,12)

# Define a linear function for use in error analysis
def linearFit(data,m,b):
    # return slope*DepVar + intercept
    return m*avgDiam + b


##### Extract Data from .csv files #####
# Loop through all 11 grind settings
for a in range(1,12): # need to start at 1 because filenames are 1-11
    # import each file, list the values, then change the lists into arrays
    file = open("data/setting%d_stats.csv" % a )
    csv_list = list(csv.reader(file))
    csv_f = np.array(csv_list)
    
    # get the value from each column in the order:  '', avg_diam, std_diam, avg_surface, std_surface, efficiency, quality
    avgDiameter.append(csv_f[:,1][1]); stdDiameter.append(csv_f[:,2][1])
    avgSurface.append(csv_f[:,3][1]); stdSurface.append(csv_f[:,4][1])
    efficiency.append(csv_f[:,5][1])
    quality.append(csv_f[:,6][1])

# Convert each array to be of type float (for error analysis)
avgDiam = np.asarray(avgDiameter, dtype='float64'); stdDiam = np.asarray(stdDiameter, dtype='float64')
avgSurf = np.asarray(avgSurface, dtype='float64'); stdSurf = np.asarray(stdSurface, dtype='float64')
eff = np.asarray(efficiency, dtype='float64'); qual = np.asarray(quality, dtype='float64')


# Calculate the average adjustment made between each whole-number grind setting and print results
for b in range(0,len(grindSetting)-1):
    settingAdjustment.append(avgDiam[b+1] - avgDiam[b])
avgAdjustment = np.sum(settingAdjustment)/len(settingAdjustment)

print("-----------------------------------------------")
print("---------Grinder Adjustment Parameters --------")
print()
print("Total Adjustment Range (Setting 11-1): {:.2} mm".format(avgDiam[-1]-avgDiam[0]))
print("Average Adjustment Between Each Setting: {:.2} mm".format(avgAdjustment))
#print("Values: {}".format(avgDiam))
print()
print("-----------------------------------------------")
print()

##### Linear Fitting #####
# This still needs some work
print("-----------------------------------------------")
print("-------------Linear Fit Parameters ------------")
fitXAxis = np.array(grindSetting).reshape(-1,1)
fitYAxis = np.array(avgDiam).reshape(-1,1)

model = LinearRegression().fit(fitXAxis,fitYAxis)

avgSizePrediction = model.predict(fitXAxis)

rSquared = model.score(fitYAxis,avgSizePrediction)

print('Intercept:', model.intercept_)
print('Slope:', model.coef_)
print('R^2 Value:', rSquared)
print("-----------------------------------------------")


input("Press Enter To Continue....")
##### Plotting #####
#Diameter#
plt.title("Average Diameter vs. Grind Setting")
plt.xlabel("Grind Setting")
plt.xlim([0,12])
plt.xticks(ticks=grindSetting)
plt.ylabel("Average Diameter [mm]")

plt.errorbar(grindSetting,avgDiam, fmt='o', color='black', ecolor='red', capsize=2, label='Average Diameter', xerr=.05, yerr=stdDiam)
plt.plot(grindSetting,avgSizePrediction, color='blue')

for i in range(0,len(avgDiam)):
    plt.annotate(avgDiam[i],(grindSetting[i]+.1*max(avgDiam),avgDiam[i]), color='black', label='diameter')
    plt.annotate(stdDiam[i],(grindSetting[i],stdDiam[i]+avgDiam[i]), color='red', label="error")

plt.show()





#Surface Area#
#fig, axs = plt.subplots(2, sharex=True, gridspec_kw={'hspace': 0})
#fig.suptitle("Surface Area vs. Grind Setting [dataset c]")
#axs[0].scatter(grindSetting,avgSurf,c='black',label="Average Surface Area")
#axs[0].set_ylabel("Average Surface Area [mm^2]",c='black')
#axs[1].scatter(grindSetting,stdSurf,c='red',label="Scatter in Surface Area")
#axs[1].set_ylabel("Scatter in Surface Area [mm^2]", c='red')
#
#for ax in axs:
#    ax.set(xlabel="Grind Setting [#]")
#    ax.label_outer()
#
#fig.legend()
#plt.show()

#Diameter#
#fig, axs = plt.subplots(2, sharex=True, gridspec_kw={'hspace': 0})
#fig.suptitle("Diameter vs. Grind Setting [dataset c]")
#axs[0].scatter(grindSetting,avgDiam,c='black',label="Average Diameter")
#axs[0].set_ylabel("Average Diameter [mm]",c='black')
#axs[0].errorbar(grindSetting,avgDiam,xerr=None,yerr=stdDiam)
#axs[1].scatter(grindSetting,stdDiam,c='red',label="Scatter in Diameter")
#axs[1].set_ylabel("Scatter in Diameter [mm]", c='red')

#for ax in axs:
#    ax.set(xlabel="Grind Setting [#]")
#    ax.label_outer()
#
#fig.legend()
#plt.show()








'''
Scratch work I don't want to delete just yet.

plt.scatter(grindSetting,avgDiameter,c='black',label="avgDiam")
plt.title("Average Diameter for Grind Setting")
plt.xlabel("Grind Setting [#]")
plt.ylabel("Grind Size [mm]")
plt.xticks(ticks=grindSetting)

plt.scatter(grindSetting,stdDiameter,c='red',label="stdDiam")
plt.title("Scatter In Diameter vs. Grind Setting")
plt.xlabel("Grind Setting [#]")
plt.ylabel("Scatter in Diameter [mm]")
plt.xticks(ticks=grindSetting)

plt.legend(loc="center right")
plt.show()

Linear Fit Stuff
#start = (1,avgDiam[0]) #best-guess of slope should be approximately 1, and starting grind size is grind setting 1.
#popt, pcov = curve_fit(linearFit, grindSetting, avgDiam, p0=start, absolute_sigma=True)
#print("\n Uncertainty in linear fit: {}".format(np.sqrt(np.diag(pcov))))
#print("\nLinear Regression Parameters:")
#print("\n m = {}".format(popt[0]))
#print("\n b = {}".format(popt[1]))
'''
