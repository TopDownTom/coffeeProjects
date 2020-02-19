import time
import math
import numpy as np
import scipy
from scipy import optimize
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import csv
import pandas as pd

# Define empty lists for each parameter & grind setting range
avgDiam=[]; stdDiam=[]
avgSurf=[]; stdSurf=[]
efficiency=[]; quality=[]
surface=[]
settingAdjustment=[]
stdDiamUpper=[]; stdDiamLower=[]
skewness=[]; kurtosis=[]
coffee_cell_size = 20

# Define functions for use in error analysis
def funcLinear(xaxis,slope,intercept):
    return slope*xaxis+intercept

def funcQuad(xaxis,a,b,c):
    return a*xaxis**2+b*xaxis+c



