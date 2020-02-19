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


def attainable_mass_simulate(volumes):
        
        #This could be done better analytically
        depth_limit = 0.1 #mm
        
        radii = (3.0/4.0*volumes/np.pi)**(1/3)
        unreachable_volumes = np.full(volumes.size, 0.0)
        
        iboulders = np.where(radii > depth_limit)
        unreachable_volumes[iboulders[0]] = 4.0/3.0*np.pi*(radii[iboulders[0]] - depth_limit)**3
        reachable_volumes = volumes - unreachable_volumes
        
        return reachable_volumes

def weighted_stddev(data, weights, frequency=False, unbiased=True):
        
        #Calculate the bias correction estimator
        if unbiased is True:
                if frequency is True:
                        bias_estimator = (np.nansum(weights) - 1.0)/np.nansum(weights)
                else:
                        bias_estimator = 1.0 - (np.nansum(weights**2))/(np.nansum(weights)**2)
        else:
                bias_estimator = 1.0

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
