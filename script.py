# ==== Load Libraries ==== # 
# directory management 
import os

# read files 
import xarray as xr
import pandas as pd
import geopandas as gpd
import rasterio as rio

# crs management 
from rasterio.crs import CRS 

# raster operations
from rasterio.mask import mask

# calculation
import numpy as np 

# visualisation 
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import string
# ============================ # 



# ==== Load Data ==== # 
# set working directory ----
clim_dir = "glacier/clim/"
dem_dir = "glacier/dems/"
thick_dir = "glacier/thick/"

# find unique glacier IDs to work on 
id = pd.read_csv("KMPG_assigned_rids_2025.csv")
rids = [id.RID_1[id["UUN"] == "B241110"].values[0], id.RID_2[id["UUN"] == "B241110"].values[0]]
# print(rids)

# get glacier outlines ----
shape = gpd.read_file("glacier_outlines/16_rgi60_LowLatitudes.shp")
shape02416 = shape[shape["RGIId"] == rids[0]]
shape01147 = shape[shape["RGIId"] == rids[1]]

print(shape02416.crs)
print(shape01147.crs) 
# glacier shapefile both epsg:4326

# Read climate + DEM + thickness data for each glacier
climate02416 = xr.open_dataset(clim_dir + "clim_rcp85_" + rids[0] + ".nc")
climate01147 = xr.open_dataset(clim_dir + "clim_rcp85_" + rids[1] + ".nc")
dem02416 = rio.open(dem_dir + "dem_" + rids[0] + ".tif")
dem01147 = rio.open(dem_dir + "dem_" + rids[1] + ".tif")
thick02416 = rio.open(thick_dir + rids[0] + "_thickness.tif")
thick01147 = rio.open(thick_dir + rids[1] + "_thickness.tif")

print(f"All glacier files loaded successfully...")
# ============================ #  


# ==== Check CRS ==== # 
print(shape02416.geometry.crs) # outline shpaefile crs is EPSG:4326
print(thick02416.crs) # EPSG:32718
print(thick01147.crs) # EPSG:32719
# rid 0: EPSG:32718
# rid 1: EPSG;32719
print(dem02416.crs)
print(dem01147.crs)
# DEMs crs relative to glacier centre
# ============================ # 


# ==== Task 1a: Plotting glacier outline over elevation ==== # 
z = dem02416.read(1) # read the first layer: elevation (m)
z1 = dem01147.read(1)
bound = dem02416.bounds # get extents 
bound1 = dem01147.bounds

# set shapefile crs to dem shapefile 
shape02416 = shape02416.to_crs(dem02416.crs)
shape01147 = shape01147.to_crs(dem01147.crs)

# plot ----
fig, ax = plt.subplots(figsize=(10, 5))
gs = GridSpec(1, 2, figure=fig)

ax.axes.set_frame_on(False)
ax.set_xticks([])
ax.set_yticks([])

ax1 = fig.add_subplot(gs[0, 0])
elevation = ax1.imshow(z, cmap="terrain", extent=(bound.left, bound.right, bound.bottom, bound.top))
cbar = fig.colorbar(elevation, shrink = .55)
cbar.set_label("Elevation (m)", fontsize=10)
cbar.set_ticks([4000, 4500, 5000, 5500])
plt.xlabel("Easting (m)", fontsize = 10)
plt.ylabel("Northing (m)", fontsize=10)
shape02416.plot(facecolor = "none", edgecolor = "black", ax=ax1)
ax1.annotate('(a)',
             xy=(-0.15, 1.15),
             xycoords='axes fraction',
             horizontalalignment='left',
             verticalalignment='top',
             fontsize=10, 
             weight="bold")

ax2 = fig.add_subplot(gs[0, 1])
elevation1 = ax2.imshow(z1, cmap = "terrain", extent = (bound1.left, bound1.right, bound1.bottom, bound1.top))
cbar = plt.colorbar(elevation1, shrink = .52)
cbar.set_label("Elevation(m)", fontsize = 10)
cbar.set_ticks([4300, 4600, 4900, 5200])
plt.xlabel("Easting(m)", fontsize = 10)
plt.ylabel("Northing(m)", fontsize = 10)
shape01147.plot(facecolor = "none", edgecolor = "black", ax=ax2)
ax2.text(-0.1, 1.1, "B")
ax2.annotate('(b)',
             xy=(-0.15, 1.1),
             xycoords='axes fraction',
             fontsize=10, 
             weight = "bold")

fig.subplots_adjust(left=0.2, wspace=0.5)
# plt.show()

fig.savefig("figure/task1a_outline_elevation.png", dpi=300)

# Get Central coordinates for each glacier for caption----
print(shape02416.CenLon)
print(shape02416.CenLat)
print(shape01147.CenLon)
print(shape01147.CenLat)
# ============================ # 



# ==== Task 1b:  Temperature and precipitation time series ====

# Step is as follows: 
# --------------
# 0. define functions 
# 1. get annual Sum of precipitation 
# 2. get annual Mean of temperature 
# 3. get moving average with an window of 11 
# 4. get standard deviation of moving average 
# 5. plot 
# --------------

# define moving average function ----
def moving_average(y, m):
 """
 2m+1 point moving average of array y.
 """
 N = y.size # number of elements in y
 mm = np.zeros(N) # make an array of zeros with the same size 
 ms = np.zeros(N) # moving standard deviation 
 for k in range(N): # creating integers up to N-1 --> k = 0, 1, 2, 3, ...N-1
     kmin = max(k-m, 0) # returns N - 1 - m 
     kmax = min(k+m, N-1) # returns N - 1 + m 
     mm[k] = np.mean(y[kmin:kmax]) # 
     ms[k] = np.std(y[kmin:kmax])
 return mm, ms

# define a function for Step. 1-4 ----
def process_climate_timeseries(ds, prcp="prcp", temp="temp", m=5): 
   """
   Input: 
    ds: xarray dataset for one glacier 
    prcp, temp: variables in ds
    m: half window for moving average (2m + 1 window)
   Returns: 
    dict with keys:
    years: 1D int array of years
    prcp: annual precipitation (mm)
    prcp_mavg: moving average (mm)
    prcp_std: moving std (mm)
    temp: mean annual temp (degC)
    temp_mavg: moving average (degC)
    temp_std: moving std (degC)
   """
   # get annual total precipitation 
   annual_prcp = ds[prcp].groupby("time.year").sum("time")
   # get annual mean temperature 
   annual_temp = ds[temp].groupby("time.year").mean("time")
   
   # return 1D np.array 
   years = annual_prcp.year.values 
   prcp = annual_prcp.values
   temp = annual_temp.values 
   
   # get moving average of 11 years 
   prcp_mavg, prcp_std = moving_average(prcp, m)
   temp_mavg, temp_std = moving_average(temp, m)

   return {
      "years": years,
      "prcp": prcp, 
      "prcp_mavg": prcp_mavg, 
      "prcp_std": prcp_std,
      "temp": temp, 
      "temp_mavg": temp_mavg, 
      "temp_std": temp_std, 
   }

# process both glaciers ----
g02416 = process_climate_timeseries(climate02416)
g01147 = process_climate_timeseries(climate01147)
climate_data = [g02416, g01147]

# plot (Step. 5)----
fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharey=False, sharex=True)

for idx, (ax, data) in enumerate(zip(axes, climate_data)): 
   years = data["years"]
   # plot precipitation 
   ax.plot(years, data["prcp_mavg"], color="blue", lw=2, label="Moving average")
   ax.fill_between(years,
                   data["prcp_mavg"] - 2*data["prcp_std"],
                   data["prcp_mavg"] + 2*data["prcp_std"],
                   color="skyblue", alpha=0.3, 
                   label = "Confidence intervals")
   ax.set_xlabel("Year", fontsize = 12)
   ax.set_ylabel("Total precipitation (mm)", fontsize = 12)


   # plot temperature 
   ax1 = ax.twinx()
   ax1.plot(years, data["temp_mavg"], color="red", lw=2, label="Moving average")
   ax1.fill_between(years,
                      data["temp_mavg"] - 2*data["temp_std"],
                      data["temp_mavg"] + 2*data["temp_std"],
                      color="lightcoral", alpha=0.2, 
                      label = "Confidence intervals")
   ax1.set_ylabel("Mean temperature (℃)", fontsize=12, rotation=270, labelpad=20, va="center")

   # capture legend handles from the first subplot so we can place unified legends
   if idx == 0:
      first_precip_handles, _ = ax.get_legend_handles_labels()
      first_temp_handles, _ = ax1.get_legend_handles_labels()
   
   # annotate 
   letter = chr(97 + idx)  # 'a' for idx=0, 'b' for idx=1
   ax.annotate(f'({letter})',
               xy=(-0.1, 1.1),
               xycoords='axes fraction',
               horizontalalignment='left',
               verticalalignment='top',
               fontsize=13,
               fontweight='bold')

# create unified legends above the first subplot (outside the figure)
# Use handles from the first subplot so colours/labels match
try:
   prec_handles = [first_precip_handles[0], first_precip_handles[1]]
   temp_handles = [first_temp_handles[0], first_temp_handles[1]]
except NameError:
   # fallback: get handles from last axes
   prec_handles, _ = axes[0].get_legend_handles_labels()
   temp_handles, _ = axes[0].twinx().get_legend_handles_labels()

# precipitation legend (left, above subplot a)
fig.legend(prec_handles, ["Moving average", "Confidence intervals"],
         loc="upper left", bbox_to_anchor=(0.2, 1), frameon=False,
         title="Precipitation (mm)")

# temperature legend (right, above subplot a)
fig.legend(temp_handles, ["Moving average", "Confidence intervals"],
         loc="upper right", bbox_to_anchor=(0.8, 1), frameon=False,
         title="Temperature (℃)")

fig.subplots_adjust(wspace=.4, top=0.88)
plt.savefig("figure/task1b_climate_timeseries.png", dpi=200)
# plt.show()
# ============================ # 

# ==== Task 1c: Calculate volume of each glacier  ====

## 02416 ----
# mask thickness tiff by glacier outline
shape02416 = shape02416.to_crs("epsg:32718")
mask_array, _ = mask(thick02416, shape02416.geometry, indexes=1, nodata=-9999)

# get the pixel area of thickness
transform = thick02416.transform 
pixelarea = np.abs(transform[0]*transform[4])
# print(pixelarea) # 25 m pixel, 625 m2 

# get glacier volume 
zp = mask_array[mask_array > 0]
volume02416 = pixelarea*np.sum(zp) 
# print(zp.max()) # 64.68 m 
# print(zp.min()) # 6.36 m
print(f"The volume of glacier 02416 as of 2018 is: " + str(np.round(volume02416, 3)) + "m3")

## 01147 ----
# mask thickness tiff by glacier outline
print(thick01147.tags())
shape01147 = shape01147.to_crs("epsg:32719")
mask_array, _ = mask(thick01147, shape01147.geometry, indexes=1, nodata=-9999)

# get glacier volume 
zp = mask_array[mask_array > 0]
volume01147 = pixelarea*np.sum(zp) 
# print(zp.max()) # 72.32 m  
# print(zp.min()) # 6.57 m
print(f"The volume of glacier 01147 as of 2018 is: " + str(np.round(volume01147, 3)) + "m3")
# ============================ # 

# ==== Task 2: Calculate the mass balance ====

# Step is as follows: 
# --------------
# 1. define functions 
#    a. get climate data in np 1d array form 
#    b. get pixel area of dem 
#    c. set shapefile crs to dem projection 
#    d. mask outside-glacier 
#    e. create empty output folder
#    f. set loops to compute glacier mass balance for each month 
# 
# 2. process both glaciers to get mass balance per month 
# 3. get annual mean mass balance 
# 4. plot annual change in glacier mass
# --------------

# 1. define functions to compute glacier mass balance----
def compute_mb(dem, shape, climate, mu = 20, lam = 0.006, Tsolid = -2): 
   """ 
   Compute glacier wide monthly mass balance 

   Inputs: 
   shape: glacier shapefile
   dem: elevation per pixel 
   climate: monthly temperature and precipitation record between 1980-2100

   Returns:
   mass_balance: 1D array of monthly change in glacier mass (kg)
    
   """
   # a. get cliamte data into np.array forms
   time = climate.time.values
   temperature = climate.temp.values
   precipitation = climate.prcp.values
   N = temperature.size

   # b. get pixel area for DEM 
   pixelArea = np.abs(dem.transform[0] * dem.transform[4])

   # c. set shapefile crs to DEM projection
   shape = shape.to_crs(dem.crs)

   # d. clip out outside-the-glacier value to be -9999
   mask_glacier, _ = mask(dem, shape.geometry, invert = False, nodata = -9999)

   # remove mask_glacier 3rd dimension 
   mask_glacier = np.squeeze(mask_glacier)

   # e. empty output folder 
   rows, cols = mask_glacier.shape # get the row and column numbers of mask_glacier 
   smb_pixel = np.zeros((rows, cols)) # create an empty 2d-array with the same rows and columns
   mass_balance = np.zeros(N)

   # f. compute mass glacier 
   for i in range(N): 
      
      temp_i = temperature[i]
      prcp_i = precipitation[i]

      # get temperature in every pixel using the temperature at the reference height, the elevation,
      # and the lapse rate
      temp_pixel = temp_i - (dem.read(1) - climate.ref_hgt) * lam

      # get melt in each pixel (a negative number) according to degree day factor
      smb_pixel[:,:] = -mu * temp_pixel

      # set melt to zero in pixels where temperature is negative or we are outside of the glacier
      smb_pixel[(mask_glacier==-9999)|(temp_pixel<0)] = 0.0

      # add the precipitation to pixels inside the glacier, only where temperature
      # is below the threshold for snow
      # (the backslash is to continue an expression on the next line)
      smb_pixel[(mask_glacier>-9999) & (temp_pixel<Tsolid)] = \
      smb_pixel[(mask_glacier>-9999) & (temp_pixel<Tsolid)] + prcp_i

      mass_balance[i] = np.sum(smb_pixel[mask_glacier != -9999]) * pixelArea

   return{
      "time" : time,
      "mass_balance_kg" : mass_balance
   }


# 2. process both glaciers 
mb02416 = pd.DataFrame(compute_mb(dem02416, shape02416, climate02416))
mb01147 = pd.DataFrame(compute_mb(dem01147, shape01147, climate01147))

print(mb02416)

# 3. get moving average of mass balance over 11 years 
ma02416, std02416 = moving_average(mb02416["mass_balance_kg"], 5)
ma01147, std01147 = moving_average(mb01147["mass_balance_kg"], 5)

# 4. plot annual change in glacier mass (kg)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

ax1.plot(mb02416.time, ma02416, lw=2, color="blue", label="Moving average")
ax1.fill_between(mb02416.time, 
                 ma02416 + 2*std02416, 
                 ma02416 - 2*std02416,
                 color="skyblue", alpha=0.3, 
                 label = "Confidence intervals")
ax1.set_xlabel("Year", fontsize=12)
ax1.set_ylabel("Mass balance (kg)", fontsize=12)
ax1.legend(frameon=False, loc="best")
ax1.annotate(f'(a)',
               xy=(-0.1, 1.1),
               xycoords='axes fraction',
               horizontalalignment='left',
               verticalalignment='top',
               fontsize=13,
               fontweight='bold')

ax2.plot(mb01147.time, ma01147, lw=2, color="red", label="Moving average")
ax2.fill_between(mb01147.time, 
                 ma01147 + 2*std01147, 
                 ma01147 - 2*std01147, 
                 color="pink", alpha=0.3, 
                 label = "Confidence intervals")
ax2.set_xlabel("Year", fontsize=12)
ax2.annotate(f'(b)',
               xy=(-0.1, 1.1),
               xycoords='axes fraction',
               horizontalalignment='left',
               verticalalignment='top',
               fontsize=13,
               fontweight='bold')

plt.show()
fig.savefig("figure/task2_mass_balance_timeseries.png", dpi=300)
# ============================ # 