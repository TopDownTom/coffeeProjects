# This module does data extraction
def dataExtraction(N,preOrPost):
    ##### Extract Data from .csv files #####
    # Loop through each grind setting
    for a in range(1,N+1): # need to start at 1 because filenames are 1-11
        # import each file, list the values, then change the lists into arrays
        statsList = pd.read_csv("ek43{}adjustmentData/setting%d_stats.csv".format(preOrPost) % a )
        statsArray = np.array(statsList)

        
        # get the value from each stats column in the order:  '', avg_diam, std_diam, avg_surface, std_surface, efficiency, quality
        avgDiam.append(float(statsArray[:,1])); stdDiam.append(float(statsArray[:,2]))
        avgSurf.append(float(statsArray[:,3])); stdSurf.append(float(statsArray[:,4]))
        efficiency.append(float(statsArray[:,5])); quality.append(float(statsArray[:,6]))

        # get the value from each .csv column in the order ID, surface, roundness, short_axis, long_axis, volume, pixel_scale
        # data for grind setting 'a' can be retrieved as parameter[a]
        settingList = pd.read_csv("ek43{}adjustmentData/setting{}.csv".format(preOrPost,a))
        settingArray = np.array(settingList)
        scale = settingArray[:,6].reshape(-1,1)
        surface.append(settingArray[:,1].reshape(-1,1)/scale**2)
        
        roundness = settingArray[:,2]/scale
        mass = settingArray[:,5]/scale**2
        volume = settingArray[:,5]/scale**3
    #    attainable_masses = attainable_mass_simulate(volume)
        diameter = 2*np.sqrt(settingArray[:,4]*settingArray[:,3])/scale
    #    diamWeight = np.max(np.ceil(attainable_masses/(coffee_cell_size/1e3)**3))
    #    diamAverage = np.sum(diameter*diamWeight/np.sum(diamWeight))
    #    data_weights = np.full(nclusters, 1)
    return statsArray, avgDiam, stdDiam, avgSurf, stdSurf, efficiency, quality, settingArray
