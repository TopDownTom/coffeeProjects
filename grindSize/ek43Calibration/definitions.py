# Define empty lists for each parameter & grind setting range
avgDiam=[]; stdDiam=[]
avgSurf=[]; stdSurf=[]
efficiency=[]; quality=[]
surface=[]
settingAdjustment=[]
stdUpper=[]; stdLower=[]

# Define functions for use in error analysis
def funcLinear(xaxis,slope,intercept):
    return slope*xaxis+intercept

def funcQuad(xaxis,a,b,c):
    return a*xaxis**2+b*xaxis+c



