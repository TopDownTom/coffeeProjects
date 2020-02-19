import time 
import numpy as np 
import math 
import pandas as pd 
import csv 
import scipy as sp 
from scipy import optimize 
from scipy.optimize import curve_fit 
from scipy import stats as stats 
import matplotlib.pyplot as plt

import '~/dataProcessingFunctions/surfaceArea'

data = np.array(pd.read_csv("setting5.csv"))
val1,val2 = surfaceArea.posNegError(data)
print("epos is {}".format(val1))
print("eneg is {}".format(val2))
