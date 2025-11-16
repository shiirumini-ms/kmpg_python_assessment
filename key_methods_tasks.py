## Task 1a) Produce an image of the DEM for one of the glaciers with the glacier outline

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

## ==== PREPARE DATA =====
# load glacier files ----
# set CRS for DEMs 
rids = ["02416", "01147"]
# crs_map = {rids[0]: 32718,  # UTM zone 18S
#           rids[1]: 32719}  # UTM zone 19S

shape02416 = gpd.read_file("myglacier/RGI60-16.02416_outline.shp")
shape01147 = gpd.read_file("myglacier/RGI60-16.01147_outline.shp")

climate02416 = xr.open_dataset("myglacier/RGI60-16.02416_climate.nc")
climate01147 = xr.open_dataset("myglacier/RGI60-16.01147_climate.nc")

# with rio.open("myglacier/RGI60-16.02416.tif", 'r+') as src:
 #             src.crs = CRS.from_epsg(crs_map[rids[0]])
dem02416 = rio.open("myglacier/RGI60-16.02416_dem.tif")

# with rio.open("myglacier/RGI60-16.01147_dem.tif", 'r+') as src:
 #           src.crs = CRS.from_epsg(crs_map[rids[1]])
dem01147 = rio.open("myglacier/RGI60-16.01147_dem.tif")

thick02416 = rio.open("myglacier/RGI60-16.02416_thickness.tif")
thick01147 = rio.open("myglacier/RGI60-16.01147_thickness.tif")
        
print("\nAll glacier files loaded successfully...")

# CRS check ----
print(shape02416.geometry.crs) # outline shpaefile crs is EPSG:4326 
print(thick02416.crs) # EPSG:32718
print(thick01147.crs) # EPSG:32719
# rid 0: EPSG:32718
# rid 1: EPSG;32719

print(dem02416.crs)  
print(dem01147.crs) 
## no crs set for both DEMs

## ==== PLOT ELEVATION DATA AND GLACIER OUTLINE =====
dem02416.bounds # extent of DEM 
z = dem02416.read(1) # read the first layer

# plot DEM with glacier outline
fig, ax = plt.subplots(figsize=(12,12))

figTop = dem02416.bounds.top 
figBottom = dem02416.bounds.bottom
figRight = dem02416.bounds.right
figLeft = dem02416.bounds.left
plt.imshow(z, cmap="jet",extent = (figLeft, figRight, figBottom, figTop))
cbar = plt.colorbar()
cbar.set_label("Elevation(m)")
cbar.set_ticks([4000, 5000, 6000])
plt.xlabel("Easting(m)")
plt.ylabel("Northing(m)")

# plot the shape file 
# shape1.plot(facecolor = "none", edgecolor = "black", ax = ax)
# each is a shapefile. 

# shape02416.to_crs(epsg=dem02416.crs, inplace=True)
print(shape02416.geometry.crs) # outline shpaefile crs is EPSG:4326 
shape02416.plot(facecolor = "none", edgecolor = "black", ax=ax)
print(dem02416.crs.to_wkt())

print(dem02416.tags())
