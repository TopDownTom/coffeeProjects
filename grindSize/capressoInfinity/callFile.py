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
#import segregatedParts2
#from segregatedParts2 import *
import grinderAnalysis

if __name__ == "__main__":

    # Ask how many grind settings were used during measurement
    N=np.int(input("Enter number of grind settings used: "))
    smallestIncrement=np.float(input("What is the smallest grind measurement increment (zero for stepped grinders)? "))
    grinderAnalysis.main(N,smallestIncrement)
   
