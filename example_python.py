# -*- coding: utf-8 -*-
"""
Created on Sat May  4 16:29:11 2024

@author: manh
"""
# =============================================================================
# insitu
# =============================================================================
import pandas as pd
import os
import common.meta as meta
import numpy as np



# Read matchup data
matchup_data_path=os.path.join('D:/work/Codes/ChlaPaper/Chla_update/TrainNN/github/Chl-CONNECT/',
                               'MDB_1990_08_28_2023_07_17_modis_l2gen.csv')
matchup_df=pd.read_csv(matchup_data_path)

matchup_df=matchup_df[~matchup_df['Comments'].isin(['Non qualifié'])]
matchup_df=matchup_df.reset_index()


# Load Rrs
bands=meta.SENSOR_BANDS['MODIS-TRAIN']
bands=[412, 443, 488, 531, 551, 667, 748]
Rrs={}
Rrs_mc={}
for band in bands:
    try:

        Rrs_mc[f'{band}']=matchup_df[f'Rrs{band}_med'].values
    except:
        pass


# Construct input
Rrs_mc_vis_nir=[Rrs_mc['412'], Rrs_mc['443'], Rrs_mc['488'], 
                Rrs_mc['531'], Rrs_mc['551'], Rrs_mc['667'], Rrs_mc['748']]
Rrs_mc_vis_nir = np.array(Rrs_mc_vis_nir).T


# Perform CONNECT algorithm
from common.Chl_CONNECT import Chl_CONNECT
Chl_NN_mc=Chl_CONNECT(Rrs_mc_vis_nir)

Chl=Chl_NN_mc.Chl_comb
Class=Chl_NN_mc.Class

# g=(matchup_df['CV_Rrs551']<0.3) & (matchup_df['Nvalid_Rrs551']>5) & (matchup_df['timediff_hour']<3)

# from common.utils import pscatter_update
# pscatter_update(matchup_df['Chla'].values[g],Chl[g],Class[g],title='MC',titlelocation='out',
#                 legend='off',
#                 legendlocation='east outside',legendcolumn=2,
#                 xlim=[1e-4,1e4],
#                 ylim=[1e-4,1e4])
