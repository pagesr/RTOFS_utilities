"""
  Prepare river climatology for GOA from
  daily GLOFAS river fields
"""
import numpy as np
import xarray

import datetime as dt
import numpy as np
from pathlib import Path
import xarray
import os
import importlib
import sys
#import pickle
import matplotlib.pyplot as plt
from yaml import safe_load

import mod_utils_ob as mutob
importlib.reload(mutob)

PPTHN = []
if len(PPTHN) == 0:
  cwd   = os.getcwd()
  aa    = cwd.split("/")
  nii   = cwd.split("/").index('python')
  PPTHN = '/' + os.path.join(*aa[:nii+1])
sys.path.append(PPTHN + '/MyPython/hycom_utils')
sys.path.append(PPTHN + '/MyPython/draw_map')
sys.path.append(PPTHN + '/MyPython')
sys.path.append(PPTHN + '/MyPython/mom6_utils')
sys.path.append('./seasonal-workflow')
from boundary import Segment
import mod_time as mtime
import mod_mom6 as mmom6



def modulo(ds):
    ds['time'] = np.arange(0, 365, dtype='float')
    ds['time'].attrs['units'] = 'days since 0001-01-01'
    ds['time'].attrs['calendar'] = 'noleap'
    ds['time'].attrs['modulo'] = ' '
    ds['time'].attrs['cartesian_axis'] = 'T'
    return ds


fconfig = 'config_goa.yaml'
with open(fconfig) as ff:
  config = safe_load(ff)

YR1   = config['climatology']['first_year']
YR2   = config['climatology']['last_year']
flriv = config['filesystem']['yearly_river_files']
years = np.arange(YR1, YR2+1)
input_files = [flriv.format(year=y) for y in years]

print(f'Reading runoff {YR1}-{YR2}')

# Note all runoff have +1 padded days for interpolation in time
rivers = xarray.open_mfdataset( input_files,
        preprocess=lambda x: x.isel(time=slice(None, 365)) )# skip padded days & leap yrs

vardata = rivers.runoff
print('Calculating climatology by day')
rivmn = vardata.groupby('time.dayofyear').mean('time').sel(dayofyear=slice(1, 365)).load()
rivmn = rivmn.rename({'dayofyear': 'time'})

# Prepare for climatological run - add attributes to
# recognize the data set as climatology during simulation
rivmn = modulo(rivmn)

# time gets inserted when using open_mfdataset
res = rivers[['area', 'lat', 'lon']].isel(time=0).drop_vars('time')
res['runoff'] = rivmn

output_dir = config['filesystem']['river_clim_path']
dfriv_out = os.path.join(output_dir, f'glofas_runoff_GOA_clim_{YR1}-{YR2}.nc')
print(f"Writing river clim --> {dfriv_out}")

res.to_netcdf(
    dfriv_out, 
    format='NETCDF3_64BIT',
    engine='netcdf4',
    unlimited_dims='time'
)

f_chck = False
if f_chck:
  j0 = 337
  i0 = 260

#  Ryr = vardata.isel(time=slice(None,365), y=j0, x=i0).load()
  Ryr = vardata.isel(y=j0, x=i0).load().data
  nyrs = int(Ryr.shape[0]/365)
  Ryr = Ryr.reshape((nyrs, 365)).transpose()

  Rav = rivmn.data[:,j0,i0].squeeze()

  plt.ion()
  fig1 = plt.figure(1,figsize=(9,8))
  plt.clf()
  ax1 = plt.axes([0.1, 0.24, 0.8, 0.7])
  ax1.plot(Ryr, color=[0.6, 0.6, 0.6])
  ax1.plot(Rav)

  ax1.set_title(f'River climatology and original GLOFAS runoff, i0={i0}, j0={j0}')


