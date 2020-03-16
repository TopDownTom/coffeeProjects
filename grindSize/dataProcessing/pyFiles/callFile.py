import time
import numpy as np
from definitions import *
import grinderAnalysis
import errorCalc

if __name__ == "__main__":

    # Ask how many grind settings were used during measurement
#    while True:
#        num=input("Enter number of grind settings used: ")
#        if float(num).is_integer():
#            N=int(num)
#            break
#        else:
#            print("N must be an integer")
    N=1

# I could turn this into the general choice which dataset we want to view (ek, capresso, vario, etc..)
#    # Do we want to see pre-adjustment or post-adjustment data?
#    while True:
#        preOrPost = input("(pre) or (post) Adjustment? ")
#        if preOrPost not in ('pre' , 'post'):
#            print("Please type 'pre' or 'post'")
#        else:
#            break
    # set a definition to not lose its place in grinderAnalysis.py
    preOrPost='pre'
    
    # Ask for smallest grind measurement increment on the grinder, used for plotting grind setting errorbars
#    while True:
#        smallestIncrement=np.float(input("What is the smallest grind measurement increment (zero for stepped grinders)? "))
#        if smallestIncrement == 0:
#            break
#        elif smallestIncrement.is_integer():
#            print("Please enter a float, not an integer")
#        else:
#            break
    smallestIncrement=0

    # Coffee cell size, this value does not seem to change the data in any places larger than hundreths. 
#    while True:
#        cellSize = input("Use default coffee cell size of 20 micrometers? (y/n)")
#        if cellSize not in ('y' , 'n'):
#            coffeeCellSize = input("Please enter a size in micrometers: ")
#        else:
#            coffeeCellSize = 20
#            break
    coffeeCellSize=20

    # Ask which data is interesting, average diameter or average surface area
#    while True:
#        type = input("Which information would you like to view? (d)iameter,(s)urface: ")
#        if type not in ('s','d'):
#            print("Incorrect Choice, please enter 's' or 'd'")
#        else:
#            break
    type='d'

    grinderAnalysis.main(N,preOrPost,smallestIncrement,coffeeCellSize,type)
