# import classification_functions as cf
# import meta
# import numpy as np
# import joblib
# import os

# class Chl_NN_2_17:
#     def __init__(self,Rrs_input: list[np.ndarray],
#                   method: str = 'pdf',
#                   distribution: str ='gamma',
#                   sensor: str ='MODIS'):
#         self.Rrs_input = Rrs_input
#         self.method = method
#         self.distribution = distribution
#         self.sensor = sensor
    
#         # Initialize inputs
#         if self.sensor=='MODIS':
#             bands=[412,443,488,531,551,667,748]
#         elif self.sensor=='OLCI':
#             bands=[412,443,490,510,560,665,709]
#         elif self.sensor=='MSI':
#             bands=[443,490,560,665,705]
            
#         if Rrs_input.ndim == 3:
#             init_shape=Rrs_input[:,:,0].shape
#             nInputs=Rrs_input.shape[2]
#             nRows=Rrs_input.shape[0]*Rrs_input.shape[1]
#             Rrs=np.full((nRows,len(bands)), np.nan)
#             for i in range(len(bands)):
#                 Rrs[:,i] = Rrs_input[:,:,i].reshape(-1)
            
#             self.p=np.full((Rrs_input.shape[0],Rrs_input.shape[1],17), np.nan)
#             self.p_n=np.full((Rrs_input.shape[0],Rrs_input.shape[1],17), np.nan)
#         elif Rrs_input.ndim == 2:
#             init_shape=Rrs_input.shape[0]
#             nInputs=Rrs_input.shape[1]
#             nRows=Rrs_input.shape[0]
#             Rrs = Rrs_input
#             self.p=np.full((nRows,17), np.nan)
#             self.p_n=np.full((nRows,17), np.nan)
            
#         if not nInputs==len(bands):
#             raise ValueError(f'Rrs input for {sensor} sensor must contain {len(bands)} components corresponding to wavelengths at: \
#                               \n {str(bands)}')


            
#         # Load NN Chl-a models
#         inputFilePath = os.path.join(os.path.dirname(__file__),'NN','LUTs',self.sensor)
#         best_model_1 = joblib.load(os.path.join(inputFilePath,'modis_model_1.pkl'))
#         best_model_2_17 = joblib.load(os.path.join(inputFilePath,'modis_model_2_17.pkl'))
        
        
#         # Classification
#         p,self.Class,_,_ = cf.Eumetsat_classif17(Rrs[:,0], Rrs[:,1], Rrs[:,2], 
#                                             Rrs[:,3], Rrs[:,4], Rrs[:,5],
#                                             sensor=self.sensor,distribution=self.distribution)
        
#         # 
#         Chl=np.full((Rrs.shape[0],2), np.nan)
#         input_2_17=np.log10(Rrs[:,0:-1])
#         mask=~np.any(np.isnan(input_2_17)|np.isinf(input_2_17), axis=1)
#         Chl[mask,0]=10.**best_model_2_17.predict(input_2_17[mask,:])

#         input_1=np.log10(Rrs)
#         mask=~np.any(np.isnan(input_1)|np.isinf(input_1), axis=1)
#         Chl[mask,1]=10.**best_model_1.predict(input_1[mask,:])
        
#         p_n=p
#         p_n[np.isnan(Rrs[:,-1]) & ~(self.Class==1) & ~(self.Class==0),0]=0
#         Chl[np.isnan(Rrs[:,-1]) & ~(self.Class==1) & ~(self.Class==0),1]=0
#         self.Chl_comb=(np.nansum(p_n[:,1:17],axis=1)*Chl[:,0] + p_n[:,0]*Chl[:,1])/np.sum(p_n,axis=1)
        
#         # Return to the initial size
#         self.Chl_comb=np.reshape(self.Chl_comb, init_shape)
        
#         if Rrs_input.ndim == 3:
#             for i in range(17):
#                 self.p[:,:,i]=np.reshape(p[:,i], init_shape)
#                 self.p_n[:,:,i]=np.reshape(p_n[:,i], init_shape)
#             self.Chl[:,:,0]=np.reshape(Chl[:,0],init_shape)
#             self.Chl[:,:,1]=np.reshape(Chl[:,1],init_shape)
#         elif Rrs_input.ndim == 2:
#             self.p=p
#             self.p_n=p_n
#             self.Chl=Chl

import common.classification_functions as cf
import common.meta
import numpy as np
import joblib
import os
import time

class Chl_NN_2_17:
    def __init__(self, Rrs_input: list[np.ndarray], 
                 method: str = 'logreg', 
                 distribution: str ='gamma', 
                 sensor: str ='MODIS',
                 logRrsNN: bool = False,
                 logRrsClassif: bool = False,
                 pTransform: bool = False):
        self.Rrs_input = Rrs_input
        self.method = method
        self.distribution = distribution
        self.sensor = sensor
        self.logRrsNN = logRrsNN
        self.logRrsClassif = logRrsClassif
        self.pTransform = pTransform
        # Define band mappings for different sensors
        bands_map = {
            'MODIS': [412, 443, 488, 531, 551, 667, 748],
            'OLCI': [412, 443, 490, 510, 560, 665, 709],
            'MSI': [443, 490, 560, 665, 705]
        }
        
        bands = bands_map.get(sensor, [])
        
        if Rrs_input.ndim not in (2, 3) or Rrs_input.shape[-1] != len(bands):
            raise ValueError(f'Rrs input for {sensor} sensor must contain {len(bands)} components corresponding to wavelengths at: {bands}')

        init_shape = Rrs_input.shape[:-1]
        Rrs = Rrs_input.reshape(-1, Rrs_input.shape[-1]) if Rrs_input.ndim == 3 else Rrs_input
        
        
        # Load NN Chl-a models
        if self.logRrsNN:
            inputFilePath = os.path.join(os.path.dirname(__file__), 'LUTs', self.sensor,'17OWTs','NN','logRrs')
            input_2_17 = np.log10(Rrs[:, :-1])
            input_1 = np.log10(Rrs)
        else:
            inputFilePath = os.path.join(os.path.dirname(__file__), 'LUTs', self.sensor,'17OWTs','NN','Rrs')
            input_2_17 = Rrs[:, :-1]
            input_1 = Rrs
            
        scaleX_2_17, scaleY_2_17, weights_and_biases_2_17 = NN_info(os.path.join(inputFilePath,'model_2_17.h5'))
        scaleX_1, scaleY_1, weights_and_biases_1 = NN_info(os.path.join(inputFilePath,'model_1.h5'))
        
        input_2_17=standardize(input_2_17, scaleX_2_17['mean'], scaleX_2_17['std'])
        input_1=standardize(input_1, scaleX_1['mean'], scaleX_1['std'])
        
        
        # Classification
        self.Class,p = cf.Eumetsat_classif17(Rrs[:,0], Rrs[:,1], Rrs[:,2], 
                                            Rrs[:,3], Rrs[:,4], Rrs[:,5],
                                            method = self.method,
                                            sensor=self.sensor,
                                            distribution=self.distribution,
                                            logRrs=self.logRrsClassif)
        if self.pTransform:
            p = np.sqrt(p) / np.sum(np.sqrt(p), axis=1, keepdims=True)

        
        # Prepare inputs for Chl predictions
        mask = ~np.isnan(input_2_17).any(axis=1) & ~np.isinf(input_2_17).any(axis=1)

        # Chl predictions
        Chl = np.full((Rrs.shape[0], 2), np.nan)
        if scaleY_2_17:
            Chl[mask, 0] = 10.**inverse_standardize(predict(weights_and_biases_2_17,input_2_17[mask]),
                                                    scaleY_2_17['mean'],scaleY_2_17['std'])
        else:
            Chl[mask, 0] = 10.**predict(weights_and_biases_2_17,input_2_17[mask])
        
        mask = ~np.isnan(input_1).any(axis=1) & ~np.isinf(input_1).any(axis=1)
        
        if scaleY_1:
            Chl[mask, 1] = 10.**inverse_standardize(predict(weights_and_biases_1,input_1[mask]),
                                                    scaleY_1['mean'],scaleY_1['std'])
        else:
            Chl[mask, 1] = 10.**predict(weights_and_biases_1,input_1[mask])
        
        # Adjustments based on p and class
        p_n = p
        invalid_mask = np.logical_and(
        np.logical_or(np.isnan(Rrs[:, -1]), Rrs[:, -1] <= 0),
        np.logical_and(~(self.Class == 1), ~(self.Class == 0))
        )
        p_n[invalid_mask, 0] = 0
        Chl[invalid_mask, 1] = 0
        self.Chl_comb = (np.nansum(p_n[:, 1:17], axis=1) * Chl[:, 0] + p_n[:, 0] * Chl[:, 1]) / np.sum(p_n, axis=1)
        
        # Reshape results to match input dimensions
        self.Chl_comb = self.Chl_comb.reshape(init_shape)
        self.Class = self.Class.reshape(init_shape)
        self.p, self.p_n = p.reshape((*init_shape, -1)), p_n.reshape((*init_shape, -1))
        if Rrs_input.ndim == 3:
            self.Chl = np.empty((*init_shape, 2))
            self.Chl[..., 0], self.Chl[..., 1] = Chl[:, 0].reshape(init_shape), Chl[:, 1].reshape(init_shape)
        else:
            self.Chl = Chl

def standardize(Z, means, stds):
    return (Z - means) / stds

def inverse_standardize(Z, means, stds):
    return Z * stds + means

def predict(weights_and_biases, X):
    for i, (weights, biases) in enumerate(weights_and_biases):
        # Matrix multiplication and add bias
        X = np.dot(X, weights) + biases
        # Apply ReLU activation function for all but the last layer
        if i < len(weights_and_biases) - 1:  # ReLU for hidden layers
            X = relu(X)
    return X.flatten()

def relu(x):
    return np.maximum(0, x)

def NN_info(inputFilePath):
    import h5py
    scaleX={}
    scaleY={}
    weights_and_biases = []
    with h5py.File(inputFilePath, 'r') as f:
        scaleX['mean'] = np.array(f['mean_X'])
        scaleX['std'] = np.array(f['std_X'])
        try:
            scaleY['mean'] = np.array(f['mean_Y'])
            scaleY['std'] = np.array(f['std_Y'])
        except:
            pass
        # Iterate through groups representing layers to load weights and biases
        for layer_name in sorted(f.keys()):
            if layer_name.startswith('layer_'):
                weights = np.array(f[layer_name]['weights'])
                biases = np.array(f[layer_name]['biases'])
                weights_and_biases.append((weights, biases))
    return scaleX, scaleY, weights_and_biases