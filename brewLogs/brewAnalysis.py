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
brewDurationMin=[]
massOverTime=[]
dirs=[]

for d in os.listdir('.'):
    if os.path.isdir(d):
        dirs.append(d)

for a, dir in enumerate(dirs):
    print('{} {}'.format(a, dirs[a]))
directoryChoice = int(input("Please Select A Number From Above For Which Directory Your Brew Data Is Stored In: "))
print()
print('Switching to directory "{}"...'.format(dirs[directoryChoice]))

os.chdir(dirs[directoryChoice])

print('The brew files in this directory are: ')
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

fig, ax = plt.subplots()
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

for a in range(0,len(brewData)):
    # Rid delimited values. Brewmaster app delimits brew mass data with ';'. Also looks out for new lines.
    brewList = re.split(';|\n', brewData[a][-1])
    # If the final entry is empty ('' or ' '), remove it.
    if not brewList.pop():
       brewList.pop()
    # Convert the string values into floats for plotting
    massOverTime.append(list(map(float, brewList)))

    # Create a linear brew time space for plotting
    brewTimes = brewData[a][6]
    brewDurationSec.append(np.linspace(0,brewTimes,len(brewList)))

    # Various plotting definitions
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mass (g)")
    ax.set_title("Mass vs. Time - {}".format(dataFiles[userChoice]))
    ax.plot(brewDurationSec[a],massOverTime[a],label = "Brew {}".format(len(brewData)-a)) 
    ax.legend()


#    textstr = '\n'.join((
#        'brew times: ' '\n',
#        'brew {} - {}s'.format(len(brewData)-a,brewTimes)
#        ))
#
#    ax.plot(brewDurationSec[a],massOverTime[a],label = "Brew {}".format(len(brewData)-a)) 
#    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,verticalalignment='top', bbox=props)





# Show the plot
plt.show()
