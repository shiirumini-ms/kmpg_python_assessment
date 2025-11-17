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
   
   # get moving average 
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

# plot ----
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

   # only plot legends on the second column 
   if idx == 1:
      # combine legends from both axes
      lines, labels = ax.get_legend_handles_labels()
      lines2, labels2 = ax1.get_legend_handles_labels()
      
      # legend for precipitation (left side)
      ax.legend([lines[0], lines[1]], ["Moving average", "Confidence intervals"], 
                loc="upper left", frameon=False, title="Precipitation (mm)")
      
      # legend for temperature (right side)
      ax1.legend([lines2[0], lines2[1]], ["Moving average", "Confidence intervals"], 
                 loc="lower right", frameon=False, title="Temperature (℃)")
   
   # annotate 
   letter = chr(97 + idx)  # 'a' for idx=0, 'b' for idx=1
   ax.annotate(f'({letter})',
               xy=(-0.1, 1.1),
               xycoords='axes fraction',
               horizontalalignment='left',
               verticalalignment='top',
               fontsize=13,
               fontweight='bold')

fig.subplots_adjust(wspace=.4)
plt.savefig("figure/task1b_climate_timeseries.png", dpi=200)
# plt.show()
# ============================ # 

# ==== Task 1c: Calculate volume of each glacier  ====
print(thick02416.tags())