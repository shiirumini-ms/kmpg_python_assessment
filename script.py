# ---- Load Libraries ---- # 
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
# ------------------------ # 

# ---- Load Data ---- # 
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
# ------------------------ # 

# ---- Check CRS ---- # 
print(shape02416.geometry.crs) # outline shpaefile crs is EPSG:4326
print(thick02416.crs) # EPSG:32718
print(thick01147.crs) # EPSG:32719
# rid 0: EPSG:32718
# rid 1: EPSG;32719
print(dem02416.crs)
print(dem01147.crs)
# DEMs crs relative to glacier centre

# ---- Task 1a: Plotting glacier outline over elevation ---- # 
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
cbar.set_label("Elevation (m)", fontsize=12)
cbar.set_ticks([4000, 4500, 5000, 5500])
plt.xlabel("Easting (m)", fontsize = 10)
plt.ylabel("Northing (m)", fontsize=10)
shape02416.plot(facecolor = "none", edgecolor = "black", ax=ax1)
ax1.annotate('(a)',
             xy=(-0.15, 1.17),
             xycoords='axes fraction',
             horizontalalignment='left',
             verticalalignment='top',
             fontsize=15)

ax2 = fig.add_subplot(gs[0, 1])
elevation1 = ax2.imshow(z1, cmap = "terrain", extent = (bound1.left, bound1.right, bound1.bottom, bound1.top))
cbar = plt.colorbar(elevation1, shrink = .52)
cbar.set_label("Elevation(m)", fontsize = 12)
cbar.set_ticks([4300, 4600, 4900, 5200])
plt.xlabel("Easting(m)", fontsize = 10)
plt.ylabel("Northing(m)", fontsize = 10)
shape01147.plot(facecolor = "none", edgecolor = "black", ax=ax2)
ax2.text(-0.1, 1.1, "B")
ax2.annotate('(b)',
             xy=(-0.15, 1.1),
             xycoords='axes fraction',
             fontsize=15)

fig.subplots_adjust(left=0.2, wspace=0.8)
# plt.show()

fig.savefig("figure/task1a_outline_elevation.png")

# Get Central coordinates for each glacier for caption----
print(shape02416.CenLon)
print(shape02416.CenLat)
print(shape01147.CenLon)
print(shape01147.CenLat)
# ------------------------ # 



# ---- Task 1b:  Temperature and precipitation time series ----
# print(climate01147)

# extract year and anom data ----
# Group by year and sum precipitation over the time dimension (within each year)
total = climate02416["prcp"].groupby("time.year").sum("time")
annualrain = total.values
years = total.year.values

# plot ----
# then moving average over 11 points 
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

# calculate a new array as an 11-point moving average of anom
mavg = moving_average(annualrain, 5)

# create figure and axes
fig, ax = plt.subplots(figsize=(8,6))

# plot total annual precipitation (moving average) with CIs
ax.plot(years, mavg[0], c = "blue", label = "Moving Average", linewidth=2)

# confidence intervals as shaded region
ax.fill_between(years, 
                mavg[0] - 2*mavg[1], 
                mavg[0] + 2*mavg[1], 
                alpha=0.3, 
                color="skyblue", 
                label="Confidence intervals") 

plt.xlabel("Year", fontsize=10)
ax.set_ylabel("Total annual precipitation (mm)", fontsize=12)
ax.legend(frameon=False, loc="best", title = "Precipitation (mm)")
plt.subplots_adjust(right = 0.7)

# get annual mean temperature
temp = climate02416["temp"].groupby("time.year").mean("time")
# total is a DataArray; extract values and year coordinate directly
annualtemp = temp.values
temp_mavg = moving_average(annualtemp, 5)

# plot temperature 
ax1 = ax.twinx()
ax1.plot(years, temp_mavg[0], c = "red", label = "Moving Average", linewidth=2)

# plot confidence intervals as shaded region
ax1.fill_between(years, 
                temp_mavg[0] - 2*temp_mavg[1], 
                temp_mavg[0] + 2*temp_mavg[1], 
                alpha=0.3, 
                color="pink", 
                label="Confidence intervals") 


# ensure secondary label sits on the right and add padding so it doesn't overlap ticks
ax1.yaxis.set_label_position("right")
ax1.yaxis.set_ticks_position("right")

ax1.set_ylabel("Monthly temperature (℃)", rotation=270, labelpad=20, va="center", fontsize=12)
# move tick labels slightly away from axis if needed
ax1.tick_params(axis="y", labelrotation=0, pad=6)
ax1.legend(frameon=False, loc='best', title = "Temperature(℃)")
plt.subplots_adjust(right = 0.7)
# plt.show()

fig.savefig("figure/task1b_climate_timeseries.png")
# ------------------------ # 

# ---- Task 1c:  Temperature and precipitation time series ----
print(thick02416.tags()) 


