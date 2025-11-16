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