#Below is code for choosing between linear or quadratic fitting, I should never need quadratic fitting for this
#coffee project so to clean the code I wanted to remove but save it here.


    ##### Fitting #####
    print()
    # Ask user which regression form to use
    #fitType = input("Which Fit Type Would You Like? (l)inear,(q)uad: ")
    fitType = 'l'
    fitTypes = [funcLinear, funcQuad]
    def fittingFunction(type):
        global fitTypePlot

        # If the type is linear, perform all linear-regression related procedures
        if type == "l":
            popt, pcov = curve_fit(funcLinear, grindSetting, data, maxfev=2000) # the regression
            perr = np.sqrt(np.diag(pcov)) # error in the regression
            ss_res = np.sum((data - funcLinear(grindSetting,*popt))**2)
            ss_tot = np.sum ((data-np.mean(data))**2)
            r_squared = 1 - (ss_res/ss_tot)
            plt.plot(grindSetting, funcLinear(grindSetting, *popt), label="Linear Fit", color='green') # eventually plots the regression against grind setting
            fitTypePlot = fitTypes[0]
            plt.text(grindSetting[0],data[-1],r'$Equation\ of\ Linear\ Fit: y={:.2}x +({:.2})$'.format(popt[0],popt[1])) # generate equation of fit on figure
            plt.text(grindSetting[0],data[9],r'$R^2={:.2}$'.format(r_squared)) # generate equation of fit on figure
            print()
            print("------------- Fit Parameters ------------")       
            print("\n Slope = {:.2} +/- {:.2}".format(popt[0],perr[1]))
            print("\n Intercept = {:.2} +/- {:.2}mm".format(popt[1],perr[0]))
            print("\n R^2 = {:.2}".format(r_squared))
            print()

        elif type == "q":
            popt, pcov = curve_fit(funcQuad, grindSetting, data, maxfev=2000)
            perr = np.sqrt(np.diag(pcov))
            plt.plot(grindSetting, funcQuad(grindSetting, *popt), label="Quadratic Fit", color='green')
            plt.text(grindSetting[0],data[10],r'$Equation\ of\ Quadratic\ Fit: y={:.2}x^2+{:.2}x+{:.2}$'.format(popt[0],popt[1],popt[2]))
            fitTypePlot = fitTypes[1]

            print()
            print("------------- Fit Parameters ------------")       
            print("\n a: {:.2} +/- {:.2}".format(popt[0],perr[0]))
            print("\n b: {:.2} +/- {:.2}".format(popt[1],perr[1]))
            print("\n c: {:.2} +/- {:.2}".format(popt[2],perr[2]))
            print()
        return popt, pcov, fitTypePlot
    fittingFunction(fitType)

