import pandas as pd
import numpy as np
from osgeo import gdal
from scipy.stats import kurtosis, skew
import multiprocessing
from functools import partial
from copy import deepcopy
from scipy.ndimage.filters import generic_filter

#%%

def mykurt(a):
    return kurtosis(a=a, nan_policy='omit')
def myskew(a):
    return skew(a=a, nan_policy='omit')

#%%

ds1 = gdal.Open(r'C:\Users\adibanpa\CloudStation\Reproj_layers\EWdeep10_clipped.tif')
ds2 = gdal.Open(r'C:\Users\adibanpa\CloudStation\Reproj_layers\EWdeep1_clipped.tif')
ds3 = gdal.Open(r'C:\Users\adibanpa\CloudStation\Reproj_layers\EWdeep2_clipped.tif')
ds4 = gdal.Open(r'C:\Users\adibanpa\CloudStation\Reproj_layers\EWdeep5_clipped.tif')
ds5 = gdal.Open(r'C:\Users\adibanpa\CloudStation\Reproj_layers\NSdeep10_clipped.tif')
ds6 = gdal.Open(r'C:\Users\adibanpa\CloudStation\Reproj_layers\NSdeep1_clipped.tif')
ds7 = gdal.Open(r'C:\Users\adibanpa\CloudStation\Reproj_layers\NSdeep2_clipped.tif')
ds8 = gdal.Open(r'C:\Users\adibanpa\CloudStation\Reproj_layers\NSdeep5_clipped.tif')
ds9 = gdal.Open(r'C:\Users\adibanpa\CloudStation\Reproj_layers\bougmask.tif')
ds10 = gdal.Open(r'C:\Users\adibanpa\CloudStation\Reproj_layers\fault10_clipped.tif')
ds11 = gdal.Open(r'C:\Users\adibanpa\CloudStation\Reproj_layers\fault1_clipped.tif')
ds12 = gdal.Open(r'C:\Users\adibanpa\CloudStation\Reproj_layers\fault2_clipped.tif')
ds13 = gdal.Open(r'C:\Users\adibanpa\CloudStation\Reproj_layers\fault5_clipped.tif')
ds14 = gdal.Open(r'C:\Users\adibanpa\CloudStation\Reproj_layers\au_reproj.tif')
ds15 = gdal.Open(r'C:\Users\adibanpa\CloudStation\Reproj_layers\psg2_rpj.tif')
ds16 = gdal.Open(r'C:\Users\adibanpa\CloudStation\Reproj_layers\psgtilt2_rpj.tif')
ds17 = gdal.Open(r'C:\Users\adibanpa\CloudStation\Reproj_layers\rtp2_rpj.tif')
ds18 = gdal.Open(r'C:\Users\adibanpa\CloudStation\Reproj_layers\rtptilt2_rpj.tif')
ds19 = gdal.Open(r'C:\Users\adibanpa\CloudStation\Reproj_layers\magAS_rpj.tif')
ds20 = gdal.Open(r'C:\Users\adibanpa\CloudStation\Reproj_layers\ewdeep3_clp.tif')
ds21 = gdal.Open(r'C:\Users\adibanpa\CloudStation\Reproj_layers\ewdeep4_clp.tif')
ds22 = gdal.Open(r'C:\Users\adibanpa\CloudStation\Reproj_layers\ewdeep6_clp.tif')
ds23 = gdal.Open(r'C:\Users\adibanpa\CloudStation\Reproj_layers\nsdeep3_clp.tif')
ds24 = gdal.Open(r'C:\Users\adibanpa\CloudStation\Reproj_layers\nsdeep4_clp.tif')
ds25 = gdal.Open(r'C:\Users\adibanpa\CloudStation\Reproj_layers\nsdeep6_clp.tif')
ds26 = gdal.Open(r'C:\Users\adibanpa\CloudStation\Reproj_layers\fault3_clp.tif')
ds27 = gdal.Open(r'C:\Users\adibanpa\CloudStation\Reproj_layers\fault4_clp.tif')
ds28 = gdal.Open(r'C:\Users\adibanpa\CloudStation\Reproj_layers\fault6_clp.tif')
#%%

#raster to array
ewdeep10 = np.array(ds1.GetRasterBand(1).ReadAsArray())
ewdeep1 = np.array(ds2.GetRasterBand(1).ReadAsArray())
ewdeep2 = np.array(ds3.GetRasterBand(1).ReadAsArray()) - ewdeep1
ewdeep3 = np.array(ds20.GetRasterBand(1).ReadAsArray()) - (ewdeep1 + ewdeep2)
ewdeep4 = np.array(ds21.GetRasterBand(1).ReadAsArray()) - (ewdeep1 + ewdeep2 + 
                  ewdeep3)
ewdeep5 = np.array(ds4.GetRasterBand(1).ReadAsArray()) - (ewdeep1 + ewdeep2 + 
                  ewdeep3 + ewdeep4)
ewdeep6 = np.array(ds22.GetRasterBand(1).ReadAsArray()) - (ewdeep1 + ewdeep2 + 
                  ewdeep3 + ewdeep4)

nsdeep10 = np.array(ds5.GetRasterBand(1).ReadAsArray())
nsdeep1 = np.array(ds6.GetRasterBand(1).ReadAsArray())
nsdeep2 = np.array(ds7.GetRasterBand(1).ReadAsArray()) - nsdeep1
nsdeep3 = np.array(ds23.GetRasterBand(1).ReadAsArray()) - (nsdeep1 + nsdeep2)
nsdeep4 = np.array(ds24.GetRasterBand(1).ReadAsArray()) - (nsdeep1 + nsdeep2 +
                  nsdeep3)
nsdeep5 = np.array(ds8.GetRasterBand(1).ReadAsArray()) - (nsdeep1 + nsdeep2 + 
                  nsdeep3 + nsdeep4)
nsdeep6 = np.array(ds25.GetRasterBand(1).ReadAsArray()) - (nsdeep1 + nsdeep2 + 
                  nsdeep3 + nsdeep4 + nsdeep5)

bouguer = np.array(ds9.GetRasterBand(1).ReadAsArray())

fault10 = np.array(ds10.GetRasterBand(1).ReadAsArray())
fault1 = np.array(ds11.GetRasterBand(1).ReadAsArray())
fault2 = np.array(ds12.GetRasterBand(1).ReadAsArray()) - fault1
fault3 = np.array(ds26.GetRasterBand(1).ReadAsArray()) - (fault1 + fault2)
fault4 = np.array(ds27.GetRasterBand(1).ReadAsArray()) - (fault1 +  fault2 + 
                 fault3)
fault5 = np.array(ds13.GetRasterBand(1).ReadAsArray()) - (fault1 +  fault2 + 
                 fault3 + fault4)
fault6 = np.array(ds28.GetRasterBand(1).ReadAsArray()) - (fault1 +  fault2 + 
                 fault3 + fault4 + fault5)

au = np.array(ds14.GetRasterBand(1).ReadAsArray())
psgrav = np.array(ds15.GetRasterBand(1).ReadAsArray())
psg_tilt = np.array(ds16.GetRasterBand(1).ReadAsArray())
rtp = np.array(ds17.GetRasterBand(1).ReadAsArray())
rtp_tilt = np.array(ds18.GetRasterBand(1).ReadAsArray())
anasig = np.array(ds19.GetRasterBand(1).ReadAsArray())


#%%
## grids of intersections

deep10 = ((ewdeep10+nsdeep10)*0.5).astype(np.int64)
deep5 = ((ewdeep5+nsdeep5)*0.5).astype(np.int64)
deep2 = ((ewdeep2+nsdeep2)*0.5).astype(np.int64)
deep1 = ((ewdeep1+nsdeep1)*0.5).astype(np.int64)
deep3 = ((ewdeep3+nsdeep3)*0.5).astype(np.int64)
deep4 =((ewdeep4+nsdeep4)*0.5).astype(np.int64)
deep6 =((ewdeep6+nsdeep6)*0.5).astype(np.int64)

#%%

x=np.arange(au.shape[1])
y=np.arange(au.shape[0])

y = np.asarray(list(y)*au.shape[1])
x = np.repeat(x,au.shape[0])

#%%

feature_list = [#ewdeep10, 
                ewdeep1, ewdeep2, ewdeep5, ewdeep3, ewdeep4, ewdeep6,
#                nsdeep10, 
                nsdeep1, nsdeep2, nsdeep5, nsdeep3, nsdeep4, nsdeep6, bouguer, 
#                deep10, 
                deep5, deep2, deep1, deep3, deep4, deep6,
#                fault10, 
#                fault1, fault2, fault3, fault4, fault6, fault5,
                au,psgrav, psg_tilt, rtp, rtp_tilt, anasig
               ]

#Determine features for training data

col = [#'ewdeep10', 
       'ewdeep1', 'ewdeep2', 'ewdeep5', 'ewdeep3', 'ewdeep4', 'ewdeep6',
       #'nsdeep10', 
       'nsdeep1', 'nsdeep2','nsdeep5', 'nsdeep3', 'nsdeep4', 'nsdeep6', 'bouguer', 
       #'deep10', 
       'deep5', 'deep2', 'deep1', 'deep3', 'deep4', 'deep6',
       #'fault10', 
#       'fault1', 'fault2', 'fault3', 'fault4', 'fault6', 'fault5', 
       'au', 'psgrav', 'psg_tilt', 'rtp', 'rtp_tilt', 'anasig'
      ]

for i, feat in enumerate(feature_list):
    
    if feat=='bouguer' or feat=='psgrav' or feat=='psg_tilt' or feat=='rtp' \
    or feat=='rtp_tilt' or feat == 'anasig':
        m = np.mean(feat[feat > -9999])
        feat[feat < -9998.0] = np.nan
    else:
        feat[feat < -9998.0] = 0.


#%%
d={'x':x, 'y':y}
import matplotlib
matplotlib.use('agg')

def wroll(npfunc, grid, winsize=(20,20)):
    #    grid = iterabs[0]
    #    npfunc = iterabs[1]
    
        P, Q = grid.shape
        N, M = winsize
        wroll_ = generic_filter(grid, function=npfunc, size=(M,N))
        wroll = wroll_[M//2:(M//2)+P-M+1, N//2:(N//2)+Q-N+1]
        return wroll



print('start')

def main():
    
    statfuncs = [np.nanmean, np.nanmin, np.nanmax, np.nanvar, np.nanmedian, 
             np.nanstd, mykurt, myskew]


#    wsize= (20,20)
#    N,M = wsize
    
    for i in range(len(col)):
        print(i)
        pool = multiprocessing.Pool(8)

        padded = np.pad(feature_list[i], (10,9), 'symmetric')
        wroll_p = partial(wroll, grid=padded)
        roll_list = pool.map(wroll_p, statfuncs)
        wroll_mean = roll_list[0]
        wroll_min = roll_list[1]
        wroll_max = roll_list[2]
        wroll_var = roll_list[3]
        wroll_med = roll_list[4]
        wroll_std = roll_list[5]
        wroll_kurt = roll_list[6]
        wroll_skew = roll_list[7]        

#        wroll_mean = np.nanmean(view_as_windows(padded, (M,N)), axis=(-2,-1))
#        wroll_min = np.nanmin(view_as_windows(padded, (M,N)),axis=(-2,-1))
#        wroll_max = np.nanmax(view_as_windows(padded, (M,N)),axis=(-2,-1))
#        wroll_var = np.nanvar(view_as_windows(padded, (M,N)),axis=(-2,-1))
#        wroll_med = np.nanmedian(view_as_windows(padded, (M,N)),axis=(-2,-1))
#        wroll_std = np.nanstd(view_as_windows(padded, (M,N)),axis=(-2,-1))
                    
        
#        P,Q = padded.shape
        
#        wroll_mean1 = generic_filter(padded, function=np.nanmean, size=(M,N))
#        wroll_mean = wroll_mean1[M//2:(M//2)+P-M+1, N//2:(N//2)+Q-N+1]
#        
#        wroll_min1 = generic_filter(padded, function=np.nanmin, size=(M,N))
#        wroll_min = wroll_min1[M//2:(M//2)+P-M+1, N//2:(N//2)+Q-N+1]
#        
#        wroll_max1 = generic_filter(padded, function=np.nanmax, size=(M,N))
#        wroll_max = wroll_max1[M//2:(M//2)+P-M+1, N//2:(N//2)+Q-N+1]
#        
#        wroll_var1 = generic_filter(padded, function=np.nanvar, size=(M,N))
#        wroll_var = wroll_var1[M//2:(M//2)+P-M+1, N//2:(N//2)+Q-N+1]
#        
#        wroll_med1 = generic_filter(padded, function=np.nanmedian, size=(M,N))
#        wroll_med = wroll_med1[M//2:(M//2)+P-M+1, N//2:(N//2)+Q-N+1]
#        
#        wroll_std1 = generic_filter(padded, function=np.nanstd, size=(M,N))
#        wroll_std = wroll_std1[M//2:(M//2)+P-M+1, N//2:(N//2)+Q-N+1]
#        
#        wroll_kurt1 = generic_filter(padded, function=mykurt, size=(M,N))
#        wroll_kurt = wroll_kurt1[M//2:(M//2)+P-M+1, N//2:(N//2)+Q-N+1]
#        
#        wroll_skew1 = generic_filter(padded, function=myskew, size=(M,N))
#        wroll_skew = wroll_skew1[M//2:(M//2)+P-M+1, N//2:(N//2)+Q-N+1]
        
        
        
        flipped = feature_list[i].T
        d[col[i]] = flipped.flatten()
        d['{0}_{1}'.format(col[i], 'mean')] = wroll_mean.T.flatten()
        d['{0}_{1}'.format(col[i], 'min')] = wroll_min.T.flatten()
        d['{0}_{1}'.format(col[i], 'max')] = wroll_max.T.flatten()
        d['{0}_{1}'.format(col[i], 'var')] = wroll_var.T.flatten()
        d['{0}_{1}'.format(col[i], 'med')] = wroll_med.T.flatten()
        d['{0}_{1}'.format(col[i], 'std')] = wroll_std.T.flatten()
        d['{0}_{1}'.format(col[i], 'kurt')] = wroll_kurt.T.flatten()
        d['{0}_{1}'.format(col[i], 'skew')] = wroll_skew.T.flatten()

if __name__ == '__main__':
    main()
    
#%%
print('done')
dataf1 = pd.DataFrame(d)
dataf1 = dataf1.dropna()
dataf2 = deepcopy(dataf1)

#dataf2.to_csv('engineered_data.csv')