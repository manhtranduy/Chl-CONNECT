

# # =============================================================================
# # Satellite image
# # =============================================================================
import os
import common.meta as meta
import numpy as np
import time
from pyhdf.SD import SD, SDC
from netCDF4 import Dataset
# Replace 'your_file.nc' with the path to your NetCDF file
folder='D:/work/Codes/Eumetsat/Combination'
file_path=os.path.join(folder,'201811010_cmems_obs-oc_glo_bgc-reflectance_my_l4-olci-300m_P1M_VN.hdf')
wls=[412,443,490,510,560,665,709]


# Open the HDF4 file
hdf = SD(file_path, SDC.READ)

# List available datasets
datasets = hdf.datasets()
for ds in datasets:
    print(ds)

# Assuming you know the dataset names
# For example, if you're looking for specific wavelengths
wls = [412, 443, 490, 510, 560, 665, 709]

sds = hdf.select('longitude')
lon=sds.get().T
sds = hdf.select('latitude')
lat=sds.get().T

for i,wl in enumerate(wls):
    variable_name = 'Rrs' + str(wl)
    if variable_name in datasets:
        sds = hdf.select(variable_name)
        data=sds.get().T
        if i==0:
            Rrs_sat=np.full((data.shape[0], data.shape[1], len(wls)), np.nan)
        Rrs_sat[:,:,i] = data  # Transpose if needed
    
# Don't forget to close the file when you're done
hdf.end()

        
start_time = time.time()
from common.Chl_CONNECT import Chl_CONNECT
# Chl_NN_sat=Chl_CONNECT(Rrs_sat,method='pdf',
#                         sensor='OLCI',
#                         pTransform=True)

Chl_NN_sat=Chl_CONNECT(Rrs_sat,method='logreg',
                       sensor='OLCI',
                       logRrsClassif=False,
                       pTransform=False)
end_time = time.time()
processing_time = end_time - start_time


Chl=Chl_NN_sat.Chl_comb
Class=Chl_NN_sat.Class

Class=np.reshape(Class,data.shape)

# Create a new NetCDF file
with Dataset('Chl_NN_OLCI_logreg_5OWTs.nc', 'w') as nc_file:
    # Define dimensions
    nc_file.createDimension('x', Chl.shape[0])
    nc_file.createDimension('y', Chl.shape[1])

    var_data = nc_file.createVariable('Chl_NN', 'f4', ('x', 'y'))
    var_data[:, :] = Chl
    var_data.units = '/mug/L'
    var_data.description = 'Chlorohyll-a'
    
    var_data = nc_file.createVariable('longitude', 'f4', ('x', 'y'))
    var_data[:, :] = lon
    var_data.description = 'longitude'
    
    var_data = nc_file.createVariable('latitude', 'f4', ('x', 'y'))
    var_data[:, :] = lat
    var_data.description = 'latitude'
    
    var_data = nc_file.createVariable('Class', 'i4', ('x', 'y'))
    var_data[:, :] = Class
    var_data.description = 'Optical Water Types'
    
# =============================================================================
# image 2
# =============================================================================
from netCDF4 import Dataset
# Replace 'your_file.nc' with the path to your NetCDF file
folder='D:/work/Codes/Eumetsat/Combination'
file_path=os.path.join(folder,'20210501-20210531_cmems_obs-oc_glo_bgc-reflectance_my_l4-olci-300m_P1M_china.nc')
wls=[412,443,490,510,560,670,709]

Rrs_sat=[]
with Dataset(file_path, mode='r') as ds:
    lon=ds.variables['lon'][:].data
    lat=ds.variables['lat'][:].data
    for i,wl in enumerate(wls):
        variable_name = 'RRS' + str(wl)
        data=ds.variables[variable_name][0,:,:].data
        Rrs_sat.append(data)
        initShape=data.shape

Rrs_sat=np.stack(Rrs_sat,2)
        

start_time = time.time()
from common.Chl_CONNECT import Chl_CONNECT
# Chl_NN_sat=Chl_CONNECT(Rrs_sat,method='pdf',
#                         sensor='OLCI',
#                         pTransform=True)

Chl_NN_sat=Chl_CONNECT(Rrs_sat,method='pdf',
                       sensor='OLCI',
                       logRrsClassif=True,
                       pTransform=False)
end_time = time.time()
processing_time = end_time - start_time


Chl=Chl_NN_sat.Chl_comb
Class=Chl_NN_sat.Class

Class=np.reshape(Class,data.shape)

with Dataset('Chl_NN_OLCI_pdf_5OWTs_China.nc', 'w') as nc_file:
    # Create dimensions
    nc_file.createDimension('lat', lat.size)
    nc_file.createDimension('lon', lon.size)
    
    # Create variables for latitude and longitude
    latitudes = nc_file.createVariable('lat', 'f4', ('lat',))
    longitudes = nc_file.createVariable('lon', 'f4', ('lon',))
    
    # Assign data to latitude and longitude variables
    latitudes[:] = lat
    longitudes[:] = lon
    
    # Create a variable for Chlorophyll data
    var_data = nc_file.createVariable('Chl_NN', 'f4', ('lat', 'lon'))
    var_data[:, :] = Chl
    var_data.units = 'μg/L'  # Correct unit symbol for micrograms per liter
    var_data.description = 'Chlorophyll-a'
    
    
    var_data = nc_file.createVariable('Class', 'i4', ('lat', 'lon'))
    var_data[:, :] = Class
    var_data.description = 'Optical Water Types'
    
    
# =============================================================================
# insitu
# =============================================================================
import pandas as pd
import os
import common.meta as meta
import numpy as np
import common.classification_functions as cf
from common.utils import pscatter_update


# =============================================================================
# Define Params
# =============================================================================
sensor = 'OLCI'
RrsInput = 'Rrs'
OWT = 5

# =============================================================================
# # Prepare Data
# =============================================================================
insitu_data_path=os.path.join('D:/work/Codes/Eumetsat/Data/','Eumetsat_Dataset_Clean.csv')

insitu_df=pd.read_csv(insitu_data_path)
# insitu_df=insitu_df.drop(insitu_df[insitu_df['Rrs_779']<=0].index)
# insitu_df=insitu_df.drop(insitu_df[insitu_df['Flagged']==1].index)
# insitu_df=insitu_df.loc[insitu_df['Flagged'] == 0]
insitu_df=insitu_df.drop(insitu_df[insitu_df['Chla']>=3000].index)

# insitu_df=insitu_df.dropna(subset=['Noisy_red','Noisy_blue', 'Baseline_shift',
#                                       'Oxygen_signal','Negative_uv_slope','QWIP_fail','Suspect'])

# MC
matchup_data_path=os.path.join('D:/work/Codes/Eumetsat/matchup/','MC_Eumetsat_Thomas.csv')
matchup_df=pd.read_csv(matchup_data_path)

# Condition 1: Filter rows where ID contains 'SOMLIT' and Comments do not contain '2', '6', or '7'
condition1 = matchup_df['ID'].str.contains('SOMLIT') & ~matchup_df['Comments'].isin(['2', '6', '7'])

# Condition 2: Filter rows where Comments contain 'Non qualifié'
condition2 = matchup_df['Comments'].str.contains('Non qualifié')

# Combine conditions with OR (|) since rows meeting either condition should be removed
combined_conditions = condition1 | condition2

# Filter the DataFrame to keep only the rows that do not meet these conditions
matchup_df = matchup_df[~combined_conditions]

# matchup_df=matchup_df.dropna(subset=['Noisy_red','Noisy_blue', 'Baseline_shift',
#                                       'Oxygen_signal','Negative_uv_slope','QWIP_fail','Suspect'])

# matchup_df=matchup_df.dropna(subset=['Noisy_red','Noisy_blue', 'Baseline_shift',
#                                       'Oxygen_signal','Negative_uv_slope','Suspect'])

# =============================================================================
# Read Rrs
# =============================================================================
olci_bands=meta.SENSOR_BANDS['OLCI']
Rrs={}
Rrs_mc={}
for band in olci_bands:
    try:
        Rrs[f'{band}']=insitu_df[f'Rrs_{band}'].values
        Rrs_mc[f'{band}']=matchup_df[f'satellite_Rrs_{band}_median_sr_1_'].values
    except:
        pass

Rrs_vis=[Rrs['412'], Rrs['443'], Rrs['490'], Rrs['510'], Rrs['560'], Rrs['665']]
Rrs_vis_nir=[Rrs['412'], Rrs['443'], Rrs['490'], Rrs['510'], Rrs['560'], Rrs['665'], Rrs['709']]
Rrs_vis = np.array(Rrs_vis).T
Rrs_vis_nir = np.array(Rrs_vis_nir).T

Rrs_mc_vis=[Rrs_mc['412'], Rrs_mc['443'], Rrs_mc['490'], Rrs_mc['510'], Rrs_mc['560'], Rrs_mc['665']]
Rrs_mc_vis_nir=[Rrs_mc['412'], Rrs_mc['443'], Rrs_mc['490'], Rrs_mc['510'], Rrs_mc['560'], Rrs_mc['665'], Rrs_mc['709']]
Rrs_mc_vis = np.array(Rrs_mc_vis).T
Rrs_mc_vis_nir = np.array(Rrs_mc_vis_nir).T

# =============================================================================
# In-situ
# =============================================================================
from common.Chl_CONNECT import Chl_CONNECT
Chl_NN=Chl_CONNECT(Rrs_vis_nir,
                   method='pdf',
                   sensor='OLCI',
                   pTransform=False)

Chl=Chl_NN.Chl_comb
Class=Chl_NN.Class
p=Chl_NN.p

pscatter_update(insitu_df['Chla'].values,Chl,Class,title='DSW',titlelocation='out',
                legend='off',
                legendlocation='east outside',legendcolumn=2,
                xlim=[1e-3,1e4],
                ylim=[1e-3,1e4])

Chl_NN_2=Chl_CONNECT(Rrs_vis_nir,
                      method='logreg',
                      sensor='OLCI',
                      spectralShift=False,
                      logRrsClassif=False,
                      pTransform=False)

Chl2=Chl_NN_2.Chl_comb
Class2=Chl_NN_2.Class
p2=Chl_NN_2.p


pscatter_update(insitu_df['Chla'].values,Chl2,Class2,title='DSW',titlelocation='out',
                legend='off',
                legendlocation='east outside',legendcolumn=2,
                xlim=[1e-3,1e4],
                ylim=[1e-3,1e4])

diff_ind=(np.isnan(Chl))&(~np.isnan(Chl2))
a=p[diff_ind,:]
b=p2[diff_ind,:]
c=Rrs_vis_nir[diff_ind,:]
C1_fil=Class[diff_ind]
C2_fil=Class2[diff_ind]
# # =============================================================================
# # MC
# # =============================================================================
Chl_NN_mc=Chl_CONNECT(Rrs_mc_vis_nir,
                      method='pdf',
                      sensor='OLCI',
                      pTransform=False,
                      spectralShift=False)

Chl_mc=Chl_NN_mc.Chl_comb
Class_mc=Chl_NN_mc.Class
p=Chl_NN_mc.p


pscatter_update(matchup_df['Chla'].values,Chl_mc,Class_mc,title='MC',titlelocation='out',
                legend='off',
                legendlocation='east outside',legendcolumn=2,
                xlim=[1e-4,1e4],
                ylim=[1e-4,1e4])

Chl_NN_mc=Chl_CONNECT(Rrs_mc_vis_nir,
                      method='logreg',
                      sensor='OLCI',
                      logRrsClassif=False,
                      pTransform=False)

Chl_mc=Chl_NN_mc.Chl_comb
Class_mc=Chl_NN_mc.Class
p=Chl_NN_mc.p


pscatter_update(matchup_df['Chla'].values,Chl_mc,Class_mc,title='MC',titlelocation='out',
                legend='off',
                legendlocation='east outside',legendcolumn=2,
                xlim=[1e-4,1e4],
                ylim=[1e-4,1e4])
