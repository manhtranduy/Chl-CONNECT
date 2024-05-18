import common.classification_functions as cf
import numpy as np
import os


class Chl_MuBR_NDCIbased:
    """
    This module (Chl_MuBR_NDCIbased.py) is responsible for combining two Neural Network models to estimate Chl-a
    
    Attributes:
        Rrs_input (list[np.ndarray]): Input remote sensing reflectance data.
        method (str): Method used for calculations, defaults to 'logreg'.
        distribution (str): Type of distribution assumed for data modeling, defaults to 'gamma'.
        sensor (str): Type of sensor used for data collection, defaults to 'OLCI'.
        spectralShift (bool): Whether spectral shifting is applied, defaults to True.
        logRrsNN (bool): If True, applies log transformation to Rrs data for neural network, defaults to False.
        logRrsClassif (bool): If True, applies log transformation to Rrs data for classification, defaults to False.
        pTransform (bool): If True, performs probability transformation on the data, defaults to False.
    
    Note:
        The default settings are tailored for OLCI nad MSI sensors.
    
    Dependencies:
        numpy, os
    
    Example:
        from Chl_MuBR_NDCIbased import Chl_MuBR_NDCIbased
        chl_mubr = Chl_MuBR_NDCIbased(Rrs_input=[Rrs412,Rrs443,Rrs490,Rrs510,Rrs560,Rrs665,Rrs709], 
                               sensor='OLCI')
        Chl = chl_mubr.Chl
        Class = chl_mubr.Class
    """
    def __init__(self, Rrs_input: list[np.ndarray], 
                 method: str = 'logreg', 
                 distribution: str ='normal', 
                 sensor: str ='OLCI',
                 spectralShift: bool = True,
                 logRrsClassif: bool = True,
                 pTransform: bool = False,
                 version:int = 1):
        self.Rrs_input = Rrs_input
        self.method = method
        self.distribution = distribution
        self.sensor = sensor
        self.logRrsClassif = logRrsClassif
        self.pTransform = pTransform
        self.spectralShift = spectralShift
        self.version = version
        # Define band mappings for different sensors
        bands_map = {
            'MODIS': [412, 443, 488, 531, 551, 667, 748],
            'OLCI': [412, 443, 490, 510, 560, 665, 709],
            'MSI': [443, 490, 560, 665, 705],
            'OLI': [443, 490, 560, 655, 865]
        }
        
        bands = bands_map.get(sensor, [])
        
        if Rrs_input.ndim not in (2, 3) or Rrs_input.shape[-1] != len(bands):
            raise ValueError(f'Rrs input for {sensor} sensor must contain {len(bands)} components corresponding to wavelengths at: {bands}')

        init_shape = Rrs_input.shape[:-1]
        Rrs = Rrs_input.reshape(-1, Rrs_input.shape[-1]) if Rrs_input.ndim == 3 else Rrs_input
        
        
        # Load Chl-a models
        input123_FilePath = os.path.join(os.path.dirname(__file__), 'LUTs', self.sensor,'5OWTs','MuBR_NDCI',
                                         f'coef123_v{self.version}.txt')
        coef123=np.loadtxt(input123_FilePath, delimiter=',')
        
        input4_FilePath = os.path.join(os.path.dirname(__file__), 'LUTs', self.sensor,'5OWTs','MuBR_NDCI',
                                         f'coef4_v{self.version}.txt')
        coef4=np.loadtxt(input4_FilePath, delimiter=',')
        
        if sensor=='OLCI':
            R1=np.log10(Rrs[:,2]/Rrs[:,1])
            R2=np.log10(Rrs[:,4]/Rrs[:,2])
            R3=np.log10(Rrs[:,5]/Rrs[:,4])
            R_NDCI= (Rrs[:,6]-Rrs[:,5])/(Rrs[:,6]+Rrs[:,5])
            input_4=R_NDCI
        if sensor=='MSI':
            R1=np.log10(Rrs[:,1]/Rrs[:,0])
            R2=np.log10(Rrs[:,2]/Rrs[:,1])
            R3=np.log10(Rrs[:,3]/Rrs[:,2])
            R_NDCI= (Rrs[:,4]-Rrs[:,3])/(Rrs[:,4]+Rrs[:,3])
            input_4=R_NDCI
        if sensor=='OLI':
            R1=np.log10(Rrs[:,1]/Rrs[:,0])
            R2=np.log10(Rrs[:,2]/Rrs[:,1])
            R3=np.log10(Rrs[:,3]/Rrs[:,2])
            R4=np.log10(Rrs[:,4]/Rrs[:,3])
            input_4=R4
        if sensor=='MODIS':
            R1=np.log10(Rrs[:,2]/Rrs[:,1])
            R2=np.log10(Rrs[:,4]/Rrs[:,2])
            R3=np.log10(Rrs[:,5]/Rrs[:,4])
            R4=np.log10(Rrs[:,6]/Rrs[:,5])
            input_4=R4
        
        input_123=np.column_stack([R1,R2,R3])
        
        

        
        # Classification
        self.Class,p = cf.classif5(Rrs[:,:6],
                                    method = self.method,
                                    sensor=self.sensor,
                                    distribution=self.distribution,
                                    logRrs=self.logRrsClassif,
                                    spectralShift=self.spectralShift)

        p=np.column_stack(p);
        
        if self.pTransform:
            p = np.sqrt(p) / np.sum(np.sqrt(p), axis=1, keepdims=True)
        
        # Prepare inputs for Chl predictions
        mask = ~np.isnan(input_123).any(axis=1) & ~np.isinf(input_123).any(axis=1)

        # Chl predictions
        Chl = np.full((Rrs.shape[0], 2), np.nan)
        Chl[mask, 0] = 10.**(coef123[0] + coef123[1]*input_123[mask,0] + 
                                          coef123[2]*input_123[mask,1] + 
                                          coef123[3]*input_123[mask,2])
        
        
        mask = ~np.isnan(input_4) & ~np.isinf(input_4)
        
        if sensor=='OLCI' or sensor =='MSI':
            Chl[mask, 1] = 10.**(coef4[0] + coef4[1]*input_4[mask] + coef4[2]*(input_4[mask]**2))
        if sensor=='OLI' or sensor =='MODIS':
            Chl[mask, 1] = 10.**(coef123[0] + coef123[1]*input_123[mask,0] + 
                                              coef123[2]*input_123[mask,1] + 
                                              coef123[3]*input_123[mask,2] +
                                              coef123[4]*input_4[mask])
        
        # Adjustments based on p and class
        p1 = np.nansum(p[:,:3],axis=1)
        p2 = np.nansum(p[:,3:5],axis=1)

        
        invalid_mask = ~np.isnan(Chl[:,0]) & np.isnan(Chl[:,1]) & ~(self.Class == 4) & ~(self.Class == 5)
        p2[invalid_mask] = 0
        Chl[invalid_mask, 1] = 0
        
        # Get rid of unrealistic values
        mask1=Chl[:,0]>15000
        p1[mask1]=0
        mask2=Chl[:,1]>15000
        p2[mask2]=0
        
        self.Chl_comb = (p1*Chl[:,0] + p2*Chl[:,1])/(p1+p2)
        
        # Reshape results to match input dimensions
        self.Chl_comb = self.Chl_comb.reshape(init_shape)
        self.Class = self.Class.reshape(init_shape)
        self.p = p.reshape((*init_shape, -1))
        if Rrs_input.ndim == 3:
            self.Chl = np.empty((*init_shape, 2))
            self.Chl[..., 0], self.Chl[..., 1] = Chl[:, 0].reshape(init_shape), Chl[:, 1].reshape(init_shape)
        else:
            self.Chl = Chl



