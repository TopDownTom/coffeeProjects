import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import math
import os

# This script has the basic functionality of overlaying mass vs. time brew plots for multiple brews of the same coffee.
# More functionality will be added over time.

# Initial list definitions
dataFiles=[]
brewDurationSec=[]
massOverTime=[]

# Find data files in $PWD of .csv filetype
for file in os.listdir("."):
    if file.endswith(".csv"):
        dataFiles.append(file)

# User selection which file to run this program on
print()
print("Index:      Filename:")
for a,files in enumerate(dataFiles):
    print(a,'          {}'.format(dataFiles[a]))
print()
userChoice = int(input("Please select a number from above for which brew you want to plot: "))
print()
print("Running file on {}...".format(dataFiles[userChoice]))

# Define the brew data 
brewData = np.array(pd.read_csv("{}".format(dataFiles[userChoice]), header=None))

for a in range(0,len(brewData)):
    # Rid delimited values. Brewmaster app delimits brew mass data with ';'. Also looks out for new lines.
    brewTimes = brewData[a][6]
    brewList = re.split(';|\n', brewData[a][-1])
    # If the final entry is empty ('' or ' '), remove it.
    if not brewList.pop():
       brewList.pop()
    # Convert the string values into floats for plotting
    massOverTime.append(list(map(float, brewList)))

    # Create a linear brew time space for plotting
    brewDurationSec.append(np.linspace(0,brewTimes,len(brewList)))

    # Various plotting definitions
    plt.xlabel("Time (s)")
    plt.ylabel("Mass (g)")
    plt.title("Mass vs. Time - {}".format(dataFiles[userChoice]))
    plt.plot(brewDurationSec[a],massOverTime[a],label = "Brew {}".format(len(brewData)-a)) 
    plt.legend()

# Show the plot
plt.show()
