

# # =============================================================================
# # Satellite image
# # =============================================================================
import os
import common.meta as meta
import numpy as np
import time
from netCDF4 import Dataset
# Replace 'your_file.nc' with the path to your NetCDF file
folder='D:/work/Codes/ChlaPaper/Chla_update/TrainNN/train'
outdir='D:/work/Codes/ChlaPaper/Chla_update/TrainNN/train/test'
file_path=os.path.join(folder,'MODIS_200307.nc')
wls=[412,443,488,531,547,667,748]

Rrs_sat=[]
with Dataset(file_path, 'r') as ds:
    lon=ds.variables['longitude'][:].data
    lat=ds.variables['latitude'][:].data
    for i,wl in enumerate(wls):
        variable_name = 'Rrs' + str(wl)
        data=ds.variables[variable_name][:].data
        Rrs_sat.append(data)
        initShape=data.shape

    Rrs_sat=np.stack(Rrs_sat,2)

        
start_time = time.time()
from common.Chl_CONNECT import Chl_CONNECT
Chl_NN_sat=Chl_CONNECT(Rrs_sat,
                       method='logreg',
                       sensor='MODIS',
                       logRrsClassif=False,
                       pTransform=False)
end_time = time.time()
processing_time = end_time - start_time


Chl=Chl_NN_sat.Chl_comb
Class=Chl_NN_sat.Class

Class=np.reshape(Class,data.shape)

# Create a new NetCDF file
with Dataset(os.path.join(outdir,'Chl_NN_MODIS_logreg_5OWTs_v2.nc'), 'w') as nc_file:
    # Create dimensions
    nc_file.createDimension('lat', lat.shape[0])
    nc_file.createDimension('lon', lat.shape[1])
    
    # Create variables for latitude and longitude
    latitudes = nc_file.createVariable('lat', 'f4', ('lat',))
    longitudes = nc_file.createVariable('lon', 'f4', ('lon',))
    
    # Assign data to latitude and longitude variables
    latitudes[:] = lat[:,0]
    longitudes[:] = lon[0,:]
    
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
sensor = 'MODIS'
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
matchup_data_path=os.path.join('D:/work/Codes/ChlaPaper/Chla_update/TrainNN/github/Chl-CONNECT/',
                               'MDB_1990_08_28_2023_07_17_modis_l2gen.csv')
matchup_df=pd.read_csv(matchup_data_path)

# Condition 1: Filter rows where ID contains 'SOMLIT' and Comments do not contain '2', '6', or '7'
condition1 = matchup_df['ID'].str.contains('SOMLIT') & ~matchup_df['Comments'].isin(['2', '6', '7'])

# Condition 2: Filter rows where Comments contain 'Non qualifié'
condition2 = matchup_df['Comments'].str.contains('Non qualifié')

# Combine conditions with OR (|) since rows meeting either condition should be removed
combined_conditions = condition1 | condition2

# Filter the DataFrame to keep only the rows that do not meet these conditions
matchup_df = matchup_df[~combined_conditions]
# matchup_df=matchup_df.loc[matchup_df['Flagged'] == 0]

matchup_df=matchup_df.reset_index()

matchup_df.rename(columns={'Rrs488_avg': 'Rrs490_avg'}, inplace=True)
matchup_df.rename(columns={'Rrs488_med': 'Rrs490_med'}, inplace=True)
matchup_df.rename(columns={'CV_Rrs488': 'CV_Rrs490'}, inplace=True)
matchup_df.rename(columns={'Nvalid_Rrs488': 'Nvalid_Rrs490'}, inplace=True)

matchup_df.rename(columns={'Rrs555_avg': 'Rrs551_avg'}, inplace=True)
matchup_df.rename(columns={'Rrs555_med': 'Rrs551_med'}, inplace=True)
matchup_df.rename(columns={'CV_Rrs555': 'CV_Rrs551'}, inplace=True)
matchup_df.rename(columns={'Nvalid_Rrs555': 'Nvalid_Rrs551'}, inplace=True)

matchup_df.rename(columns={'Rrs667_avg': 'Rrs665_avg'}, inplace=True)
matchup_df.rename(columns={'Rrs667_med': 'Rrs665_med'}, inplace=True)
matchup_df.rename(columns={'CV_Rrs667': 'CV_Rrs665'}, inplace=True)
matchup_df.rename(columns={'Nvalid_Rrs667': 'Nvalid_Rrs665'}, inplace=True)

# matchup_df=matchup_df.dropna(subset=['Noisy_red','Noisy_blue', 'Baseline_shift',
#                                       'Oxygen_signal','Negative_uv_slope','QWIP_fail','Suspect'])

# matchup_df=matchup_df.dropna(subset=['Noisy_red','Noisy_blue', 'Baseline_shift',
#                                       'Oxygen_signal','Negative_uv_slope','Suspect'])

# =============================================================================
# Read Rrs
# =============================================================================
bands=meta.SENSOR_BANDS['MODIS-TRAIN']
bands=[412, 443, 490,510, 531, 551, 665, 748]
Rrs={}
Rrs_mc={}
for band in bands:
    try:
        Rrs[f'{band}']=insitu_df[f'Rrs_{band}'].values
        Rrs_mc[f'{band}']=matchup_df[f'Rrs{band}_med'].values
    except:
        pass
mask=np.logical_or(np.isnan(insitu_df['Rrs_551'].values), insitu_df['Rrs_551'].values == 0)
Rrs['551'][mask]=insitu_df['Rrs_560'].values[mask]

Rrs_vis=[Rrs['412'], Rrs['443'], Rrs['490'], Rrs['531'], Rrs['551'], Rrs['665']]
Rrs_vis_nir=[Rrs['412'], Rrs['443'], Rrs['490'], Rrs['531'], Rrs['551'], Rrs['665'], Rrs['748']]
Rrs_vis = np.array(Rrs_vis).T
Rrs_vis_nir = np.array(Rrs_vis_nir).T

Rrs_mc_vis=[Rrs_mc['412'], Rrs_mc['443'], Rrs_mc['490'], Rrs_mc['531'], Rrs_mc['551'], Rrs_mc['665']]
Rrs_mc_vis_nir=[Rrs_mc['412'], Rrs_mc['443'], Rrs_mc['490'], Rrs_mc['531'], Rrs_mc['551'], Rrs_mc['665'], Rrs_mc['748']]
Rrs_mc_vis = np.array(Rrs_mc_vis).T
Rrs_mc_vis_nir = np.array(Rrs_mc_vis_nir).T

# =============================================================================
# In-situ
# =============================================================================
from common.Chl_CONNECT import Chl_CONNECT
# Chl_NN=Chl_CONNECT(Rrs_vis_nir,
#                    method='pdf',
#                    sensor='MODIS',
#                    pTransform=False)

# Chl=Chl_NN.Chl_comb
# Class=Chl_NN.Class
# p=Chl_NN.p

# pscatter_update(insitu_df['Chla'].values,Chl,Class,title='DSW',titlelocation='out',
#                 legend='off',
#                 legendlocation='east outside',legendcolumn=2,
#                 xlim=[1e-3,1e4],
#                 ylim=[1e-3,1e4])

Chl_NN_2=Chl_CONNECT(Rrs_vis_nir,
                      method='logreg',
                      sensor='MODIS',
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

# diff_ind=(np.isnan(Chl))&(~np.isnan(Chl2))
# a=p[diff_ind,:]
# b=p2[diff_ind,:]
# c=Rrs_vis_nir[diff_ind,:]
# C1_fil=Class[diff_ind]
# C2_fil=Class2[diff_ind]
# # =============================================================================
# # MC
# # =============================================================================
# Chl_NN_mc=Chl_CONNECT(Rrs_mc_vis_nir,
#                       method='pdf',
#                       sensor='MODIS',
#                       pTransform=False,
#                       spectralShift=False)

# Chl_mc=Chl_NN_mc.Chl_comb
# Class_mc=Chl_NN_mc.Class
# p=Chl_NN_mc.p


# pscatter_update(matchup_df['Chla'].values,Chl_mc,Class_mc,title='MC',titlelocation='out',
#                 legend='off',
#                 legendlocation='east outside',legendcolumn=2,
#                 xlim=[1e-4,1e4],
#                 ylim=[1e-4,1e4])

Chl_NN_mc=Chl_CONNECT(Rrs_mc_vis_nir,
                      method='logreg',
                      sensor='MODIS',
                      logRrsClassif=False,
                      pTransform=False)

Chl_mc=Chl_NN_mc.Chl_comb
Class_mc=Chl_NN_mc.Class
p=Chl_NN_mc.p
g=(matchup_df['CV_Rrs551']<0.2) & (matchup_df['Nvalid_Rrs551']>5) & (matchup_df['timediff_hour']<2)


pscatter_update(matchup_df['Chla'].values[g],Chl_mc[g],Class_mc[g],title='MC',titlelocation='out',
                legend='off',
                legendlocation='east outside',legendcolumn=2,
                xlim=[1e-4,1e4],
                ylim=[1e-4,1e4])
