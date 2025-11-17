# import libraries ----
# load libraries ----
import os 
import xarray as xr 
import numpy as np 
import pandas as pd 
import geopandas as gpd
import rasterio as rio 
import matplotlib.pyplot as plt 
from rasterio.features import geometry_mask  
from rasterio.crs import CRS

# specify glacier RGI ID 
rids = ['RGI60-16.02416', "RGI60-16.01147"]

# read in the DEM as an array named "z"
dem02416 = rio.open('glacier/dems/dem_' + rids[0] + '.tif')
z = dem02416.read(1) 

# use the dem02416.transform array to find cell area in m^2
# cell_area = np.abs(dem02416.transform[0]*dem02416.transform[4])

# read the RGI shapefile and set gdf to a new dataframe containing only 1 row
shape = gpd.read_file('glacier_outlines/16_rgi60_LowLatitudes.shp')
ind = shape.RGIId.str.fullmatch(rids[0])
shape02416 = shape[ind]
# print(shape02416)

# transform glacier outline from lat-lon to the UTM coordinate system of the DEM
shape02416 = shape02416.to_crs(dem02416.crs)

# plot DEM with glacier outline 
fig, ax = plt.subplots(figsize = (10, 10))

figTop = dem02416.bounds.top 
figBottom = dem02416.bounds.bottom
figRight = dem02416.bounds.right
figLeft = dem02416.bounds.left
plt.imshow(z, cmap="terrain",extent = (figLeft, figRight, figBottom, figTop))
cbar = plt.colorbar(shrink=.9)
cbar.set_label("Elevation(m)", fontsize = 14)
cbar.set_ticks([4000, 5000, 6000])
plt.xlabel("Easting(m)", fontsize = 15)
plt.ylabel("Northing(m)", fontsize = 15)
shape02416.plot(facecolor = "none", edgecolor = "black", ax=ax)
plt.show()


# Repeat the same task for RGI ID "RGI60-16.01147"
# read in the DEM as an array named "z"
dem01147 = rio.open('glacier/dems/dem_' + rids[1] + '.tif')
z1 = dem01147.read(1) 

# use the dem02416.transform array to find cell area in m^2
# cell_area = np.abs(dem02416.transform[0]*dem02416.transform[4])

# read the RGI shapefile and set gdf to a new dataframe containing only 1 row
shape = gpd.read_file('glacier_outlines/16_rgi60_LowLatitudes.shp')
ind = shape.RGIId.str.fullmatch(rids[1])
shape01147 = shape[ind]
# print(shape01147)

# transform glacier outline from lat-lon to the UTM coordinate system of the DEM
shape01147 = shape01147.to_crs(dem01147.crs)

# plot DEM with glacier outline 
fig, ax = plt.subplots(figsize = (10, 10))

figTop = dem01147.bounds.top 
figBottom = dem01147.bounds.bottom
figRight = dem01147.bounds.right
figLeft = dem01147.bounds.left
plt.imshow(z1, cmap="terrain",extent = (figLeft, figRight, figBottom, figTop))
cbar = plt.colorbar(shrink=.9)
cbar.set_label("Elevation(m)", fontsize = 14)
cbar.set_ticks([4500, 4800, 5100])
plt.xlabel("Easting(m)", fontsize = 15)
plt.ylabel("Northing(m)", fontsize = 15)
shape01147.plot(facecolor = "none", edgecolor = "black", ax=ax)
plt.show()

print(z1.max())
print(z1.min())

# calculate a new array as an 11-point moving average of anom
mavg = moving_average(annualrain, 5)

plt.clf()
prcp01147 = climate01147["prcp"].groupby("time.year").sum("time")
prcp01147.plot()
plt.show()
# plot ----
fig, ax = plt.subplots(figsize=(7, 5))

# plot rain 
ax.plot(years, mavg[0], c = "blue", label = "Moving Average", linewidth=2)

# confidence intervals as shaded region
ax.fill_between(years, 
                mavg[0] - 2*mavg[1], 
                mavg[0] + 2*mavg[1], 
                alpha=0.3, 
                color="skyblue", 
                label="Confidence intervals") 

plt.xlabel("Year", fontsize=10)
ax.set_ylabel("Total precipitation (mm)", fontsize=12)
ax.legend(frameon=False, loc="best", title = "Precipitation (mm)")
ax.annotate('(a)',
             xy=(-0.1, 1.1),
             xycoords='axes fraction',
             horizontalalignment='left',
             verticalalignment='top',
             fontsize=13)
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

ax1.set_ylabel("Mean temperature (℃)", rotation=270, labelpad=20, va="center", fontsize=12)
# move tick labels slightly away from axis if needed
ax1.tick_params(axis="y", labelrotation=0, pad=6)
ax1.legend(frameon=False, loc='best', title = "Temperature(℃)")
plt.subplots_adjust(right = 0.7)
# plt.show()

# fig.savefig("figure/task1b_climate_timeseries.png")