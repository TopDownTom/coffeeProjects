#Weighted standard deviation (un-biased reliability weights by default)
def weighted_stddev(self, data, weights, frequency=False, unbiased=True):
        
        #Calculate the bias correction estimator
        if unbiased is True:
                if frequency is True:
                        bias_estimator = (np.nansum(weights) - 1.0)/np.nansum(weights)
                else:
                        bias_estimator = 1.0 - (np.nansum(weights**2))/(np.nansum(weights)**2)
        else:
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

#Method to update statistics of the PSD
def update_statistics(self):
        #self.eff_var,self.ey_stddev_var,self.ey_average_var,self.surf_stddev_var,self.surf_average_var,self.diam_stddev_var,self.diam_average_var,
        
        #Verify that clusters were defined
        if self.nclusters is None:
                return
        
        #Read internal data
        try:
                pixel_scale = float(self.pixel_scale_var.get())
        except:
                return
        
        diameters = 2*np.sqrt(self.clusters_long_axis*self.clusters_short_axis)/pixel_scale
        surfaces = self.clusters_surface/pixel_scale**2
        volumes = self.clusters_volume/pixel_scale**3
        eys = self.ey_simulate(surfaces)
        attainable_masses = self.attainable_mass_simulate(volumes)
        effs = attainable_masses/volumes
        
        #diameters_average = np.mean(diameters)
        #diameters_stddev = np.std(diameters)
        weights = np.maximum(np.ceil(attainable_masses/(coffee_cell_size/1e3)**3),1)
        diameters_average = np.sum(diameters*weights)/np.sum(weights)
        diameters_stddev = self.weighted_stddev(diameters,weights,frequency=True,unbiased=True)
        
        #surfaces_average = np.mean(surfaces)
        #surfaces_stddev = np.std(surfaces)
        weights = np.maximum(np.ceil(attainable_masses/(coffee_cell_size/1e3)**3),1)
        surfaces_average = np.sum(surfaces*weights)/np.sum(weights)
        surfaces_stddev = self.weighted_stddev(surfaces,weights,frequency=True,unbiased=True)
        quality = surfaces_average/surfaces_stddev
        
        #volumes_average = np.mean(volumes)
        #volumes_stddev = np.std(volumes)
        
        weights = eys*attainable_masses
        eys_average = np.sum(eys*weights)/np.sum(weights)*100
        eys_stddev = self.weighted_stddev(eys,weights,frequency=True,unbiased=True)*100
        
        effs_average = np.mean(effs)*100
        
        diameters_average_str = "{0:.{1}f}".format(diameters_average, 2)
        diameters_stddev_str = "{0:.{1}f}".format(diameters_stddev, 2)
        
        surfaces_average_str = "{0:.{1}f}".format(surfaces_average, 2)
        surfaces_stddev_str = "{0:.{1}f}".format(surfaces_stddev, 2)
        
        #volumes_average_str = "{0:.{1}f}".format(volumes_average, 3)
        #volumes_stddev_str = "{0:.{1}f}".format(volumes_stddev, 3)
        
        eys_average_str = "{0:.{1}f}".format(eys_average, 1)
        eys_stddev_str = "{0:.{1}f}".format(eys_stddev, 1)
        
        effs_average_str = "{0:.{1}f}".format(effs_average, 1)
        q_str = "{0:.{1}f}".format(quality, 2)
        
        self.diam_average_var.set(diameters_average_str)
        self.diam_stddev_var.set(diameters_stddev_str)
        
        self.surf_average_var.set(surfaces_average_str)
        self.surf_stddev_var.set(surfaces_stddev_str)
        
        #self.ey_average_var.set(eys_average_str)
        #self.ey_stddev_var.set(eys_stddev_str)
        
        self.eff_var.set(effs_average_str)
        self.q_var.set(q_str)

#Method to erase clusters
def erase_clusters(self, event):
        
        #Verify that clusters were defined
        if self.nclusters is None:
                
                #Update the user interface status
                self.status_var.set("Coffee Particles not Detected Yet... Use Launch Particle Detection Button...")
                
                #Update the user interface
                self.master.update()
                
                #Return to caller
                return
        
        #Verify that the cluster image is being displayed
        if self.display_type.get() != outlines_image_display_name:
                
                #Update the user interface status
                self.status_var.set("Please select 'Display Type' = 'Cluster Outlines' to use the 'Erase Clusters' tool.")
                
                #Update the user interface
                self.master.update()
                
                return
        
        #If "Erase Clusters" mode was off
        if self.erase_clusters_mode is False:
                
                #Update the user interface status
                self.status_var.set("Entered 'Erase Clusters' mode. Click with the mouse to erase all clusters within the circle. Zoom in or out to set precision. Hit Escape or the 'Erase Clusters' button to end.")
                
                #Update the user interface
                self.master.update()
                
                #Update "Erase Clusters" internal status and return
                self.erase_clusters_mode = True
                
                return
        
        #If the 'Erase Clusters' mode was selected then this is what needs to be quit
        if self.erase_clusters_mode is True:
                
                #Update the user interface status
                self.status_var.set("The 'Erase Clusters' mode was deactivated.")
                
                #Update the user interface
                self.master.update()
                
                #Update "Erase Clusters" internal status and return
                self.erase_clusters_mode = False
                
                return
        
        
#Method to create histogram
def create_histogram(self, event):
        
        #Verify that clusters were defined
        if self.nclusters is None:
                
                #Update the user interface status
                self.status_var.set("Coffee Particles not Detected Yet... Use Launch Particle Detection Button...")
                
                #Update the user interface
                self.master.update()
                
                #Return to caller
                return
        
        #Quit "Erase Clusters" mode if it was still on
        self.erase_clusters_mode = False
        
        #Check that a physical size was set
        if self.pixel_scale_var.get() == "None":
                
                #Update the user interface status
                self.status_var.set("The Pixel Scale Has Not Been Defined Yet... Use the Left Mouse Button to Draw a Line on a Reference Object and Choose a Reference Length in mm...")
                
                #Update the user interface
                self.master.update()
                
                #Return to caller
                return
        
        #Size of figure in pixels
        figsize_pix = (self.canvas_width, self.canvas_height)
        
        #Transform these in pixels with the default DPI value
        my_dpi = 192
        figsize_inches = (figsize_pix[0]/my_dpi, figsize_pix[1]/my_dpi)
        
        #Prepare a figure to display the plot
        fig = plt.figure(figsize=figsize_inches, dpi=my_dpi)
        
        # === Generate histogram from data ===
        
        bins_input, ypos_errorbar = self.psd_hist_from_data(self)
        
        #If comparison data is loaded plot it
        if self.comparison.nclusters is not None:
                self.psd_hist_from_data(self.comparison, hist_color=[74, 124, 179], hist_label=self.comparison_data_label_var.get(), bins_input=bins_input, histtype="step", ypos_errorbar=ypos_errorbar*2)
        
        #Make xlog if needed
        if self.xlog_var.get() == 1:
                plt.xscale("log")
        
        #Set X and Y labels
        plt.xlabel(self.xlabel, fontsize=16)
        plt.ylabel(self.ylabel, fontsize=16)
        
        #Add legend
        plt.legend(loc=self.legend_type.get().lower())
        
        # Change size and font of tick labels
        ax = plt.gca()
        
        #Set the label fonts
        minortick_fontsize = 5
        majortick_fontsize = 8
        plt.tick_params(axis='both', which='major', labelsize=majortick_fontsize)
        
        #Make ticks longer and thicker
        ax.tick_params(axis="both", length=5, width=2, which="major")
        ax.tick_params(axis="both", length=4, width=1, which="minor")
        
        #Use a tight layout
        plt.tight_layout()
        
        #In xlog mode do not use scientific notation
        if self.xlog_var.get() == 1:
                ax.xaxis.set_minor_formatter(mticker.LogFormatter())
                ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        
        # === Transform the figure to a PIL object ===
        
        #Draw the figure in the buffer
        fig.canvas.draw()
        
        #Tranform the figure in a numpy array
        figdata = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        figdata = figdata.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        #Read the shape of the numpy array
        fw, fh, fd = figdata.shape
        
        #Transform numpy array in a PIL image
        self.img_histogram = Image.frombytes("RGB", (fh, fw), figdata)
        
        #Set the cluster map image as the currently plotted object
        self.display_type.set(histogram_image_display_name)
        self.img = self.img_histogram
        
        #Remove the no label image if no image was loaded yet
        if self.img_source is None:
                self.noimage_label.pack_forget()
        
        #Refresh the image that is displayed
        self.redraw(x=self.last_image_x, y=self.last_image_y)
        
        #Refresh the user interface status
        self.status_var.set("Histogram Image is Now Displayed...")
        
        #Refresh the state of the user interface window
        self.master.update()

#Method to load data from disk
def load_data(self, event):
        
        #Update root to avoid problems with file dialog
        self.master.update()
        
        #Invoke a file dialog to select data file
        csv_data_filename = filedialog.askopenfilename(initialdir=self.output_dir,title="Select a CSV data file",filetypes=(("csv files","*.csv"),("all files","*.*")))
        
        #Create a Pandas dataframe from the CSV data
        dataframe = pd.read_csv(csv_data_filename)
        
        #Ingest data in system variables
        self.clusters_surface = dataframe["SURFACE"].values
        self.clusters_roundness = dataframe["ROUNDNESS"].values
        self.clusters_long_axis = dataframe["LONG_AXIS"].values
        self.clusters_short_axis = dataframe["SHORT_AXIS"].values
        self.clusters_volume = dataframe["VOLUME"].values
        self.nclusters = self.clusters_surface.size
        self.pixel_scale_var.set(str(dataframe["PIXEL_SCALE"].values[0]))
        
        #Update the user interface status
        self.status_var.set("Data Loaded into Memory...")
        
        #Refresh histogram
        self.create_histogram(None)
        
#Method to load comparison data from disk
def load_comparison_data(self, event):
        
        #Update root to avoid problems with file dialog
        self.master.update()
        
        #Invoke a file dialog to select data file
        csv_data_filename = filedialog.askopenfilename(initialdir=self.output_dir,title="Select a CSV data file",filetypes=(("csv files","*.csv"),("all files","*.*")))
        
        #Create a Pandas dataframe from the CSV data
        dataframe = pd.read_csv(csv_data_filename)
        
        #Ingest data in system variables
        self.comparison.nclusters = dataframe["SURFACE"].values.size
        self.comparison.clusters_surface = dataframe["SURFACE"].values
        self.comparison.clusters_roundness = dataframe["ROUNDNESS"].values
        self.comparison.clusters_long_axis = dataframe["LONG_AXIS"].values
        self.comparison.clusters_short_axis = dataframe["SHORT_AXIS"].values
        self.comparison.clusters_volume = dataframe["VOLUME"].values
        
        self.comparison.pixel_scale_var = StringVar()
        self.comparison.pixel_scale_var.set(str(dataframe["PIXEL_SCALE"].values[0]))
        
        #Activate the comparison data label
        self.comparison_data_label_id.config(state=NORMAL)
        self.simple_comparison_data_label_id.config(state=NORMAL)
        
        #If there is already a histogram in play, refresh it
        if self.img_histogram is not None:
                self.create_histogram(None)
        
        #Update the user interface status
        self.status_var.set("Comparison Data Loaded into Memory...")

#Method to flush comparison data
def flush_comparison_data(self):
        
        #Reset system variables
        self.comparison.clusters_surface = None
        self.comparison.clusters_roundness = None
        self.comparison.clusters_long_axis = None
        self.comparison.clusters_short_axis = None
        self.comparison.clusters_volume = None
        self.comparison.nclusters = None
        self.comparison.pixel_scale_var.set(None)
        
        #Deactivate the comparison data label
        self.comparison_data_label_id.config(state=DISABLED)
        self.simple_comparison_data_label_id.config(state=DISABLED)
        
        #If there is already a histogram in play, refresh it
        if self.img_histogram is not None:
                self.create_histogram(None)
        
        #Update the user interface status
        self.status_var.set("Comparison Data Flushed from Memory...")
        
#Method to save data to disk
def save_data(self, event):
        
        #Verify if PSD analysis was done
        if self.nclusters is None:
                
                #Update the user interface status
                self.status_var.set("Particles not Detected Yet... Use Launch Particle Detection Button...")
                
                #Update the user interface
                self.master.update()
                
                #Return to caller
                return
        
        #Read internal data
        try:
                pixel_scale = float(self.pixel_scale_var.get())
        except:
                #Update the user interface status
                self.status_var.set("Some Options in the User Interface are Invalid Numbers...")
                
                #Update the user interface
                self.master.update()
                
                #Return to caller
                return
        
        #Create a Pandas dataframe for easier saving
        dataframe = pd.DataFrame({"SURFACE":self.clusters_surface,"ROUNDNESS":self.clusters_roundness,"SHORT_AXIS":self.clusters_short_axis,"LONG_AXIS":self.clusters_long_axis,"VOLUME":self.clusters_volume,"PIXEL_SCALE":pixel_scale})
        dataframe.index.name = "ID"
        
        #If expert mode is off ask for an output directory
        filename = self.session_name_var.get()+"_data.csv"
        if self.expert_mode is True:
                full_filename = self.output_dir+os.sep+filename
        else:
                full_filename = filedialog.asksaveasfilename(initialdir=self.output_dir, initialfile=filename, title="Select an output file name")
        
        #Create a Pandas dataframe for stats
        stats_dataframe = pd.DataFrame({"AVG_DIAM":[float(self.diam_average_var.get())],"STD_DIAM":[float(self.diam_stddev_var.get())], "AVG_SURF":[float(self.surf_average_var.get())],"STD_SURF":[float(self.surf_stddev_var.get())], "EFF":[float(self.eff_var.get())],"QUAL":[float(self.q_var.get())]})
        
        #Save files to CSV
        dataframe.to_csv(full_filename)
        
        #Save stats too
        stats_filename = os.path.dirname(full_filename)+os.sep+os.path.splitext(os.path.basename(full_filename))[0]+"_stats.csv"
        stats_dataframe.to_csv(stats_filename)
        
        #Update the user interface status
        self.status_var.set("Data Saved to "+filename+"...")
        
        #Update the user interface
        self.master.update()

#Method to save figure to disk
def save_histogram(self, event):
        
        #Verify that a figure exists
        if self.img is None:
                
                #Update the user interface status
                self.status_var.set("No View Created Yet...")
                
                #Update the user interface
                self.master.update()
                
                #Return to caller
                return
        
        #Update the user interface status
        self.status_var.set("Saving View...")
        
        #Update the user interface
        self.master.update()
        
        #If display is source image
        image_code = ""
        if self.display_type.get() == original_image_display_name:
                image_code = "source_image"
        
        #If display is threshold image
        if self.display_type.get() == threshold_image_display_name:
                image_code = "threshold_image"
        
        #If display is outlines image
        if self.display_type.get() == outlines_image_display_name:
                image_code = "outlines_image"
        
        #If display is histogram
        if self.display_type.get() == histogram_image_display_name:
                #Determine filename code for this type of histogram
                ihist = np.where(np.array(self.hist_choices) == self.histogram_type.get())
                image_code = self.hist_codes[ihist[0][0]]
        
        #If expert mode is off ask for an output directory
        filename = self.session_name_var.get()+"_hist_"+image_code+".png"
        if self.expert_mode is True:
                full_filename = self.output_dir+os.sep+filename
        else:
                full_filename = filedialog.asksaveasfilename(initialdir=self.output_dir, initialfile=filename, title="Select an output file name")
        
        #Save file to PNG
        self.img.save(full_filename)
        
        #Update the user interface status
        self.status_var.set("Current View Saved to "+filename+"...")
        
        #Update the user interface
        self.master.update()

#Method to quit reset interface
def reset_gui(self):
        python = sys.executable
        os.execl(python, python, * sys.argv)

#Method to quit user interface
def quit_gui(self):
        root.quit()

#Method to display help
def launch_help(self):
        webbrowser.open("https://www.dropbox.com/s/m2af0aer2e17xie/coffee_grind_size_manual.pdf?dl=0")
