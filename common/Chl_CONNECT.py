import common.classification_functions as cf
import numpy as np
import os


class Chl_CONNECT:
    """
    This module (Chl_CONNECT.py) is responsible for combining two Neural Network models to estimate Chl-a
    
    Attributes:
        Rrs_input (list[np.ndarray]): Input remote sensing reflectance data.
        method (str): Method used for calculations, defaults to 'logreg'.
        distribution (str): Type of distribution assumed for data modeling, defaults to 'gamma'.
        sensor (str): Type of sensor used for data collection, defaults to 'MODIS'.
        spectralShift (bool): Whether spectral shifting is applied, defaults to True.
        logRrsNN (bool): If True, applies log transformation to Rrs data for neural network, defaults to False.
        logRrsClassif (bool): If True, applies log transformation to Rrs data for classification, defaults to False.
        pTransform (bool): If True, performs probability transformation on the data, defaults to False.
    
    Note:
        The default settings are tailored for MODIS sensor data but can be adjusted for other sensors.
    
    Dependencies:
        numpy, os
    
    Example:
        from Chl_CONNECT import Chl_CONNECT
        chl_conn = Chl_CONNECT(Rrs_input=[Rrs412,Rrs443,Rrs488,Rrs531,Rrs551,Rrs667,Rrs748], 
                               sensor='MODIS')
        Chl = chl_conn.Chl
        Class = chl_conn.Class
    """
    def __init__(self, Rrs_input: list[np.ndarray], 
                 method: str = 'logreg', 
                 distribution: str ='gamma', 
                 sensor: str ='MODIS',
                 spectralShift: bool = True,
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
        self.spectralShift = spectralShift
        # Define band mappings for different sensors
        bands_map = {
            'MODIS': [412, 443, 488, 531, 551, 667, 748],
            'OLCI': [412, 443, 490, 510, 560, 665, 709],
            'MSI': [443, 490, 560, 665, 705]
        }
        
        bands = bands_map.get(sensor, [])
        if isinstance(Rrs_input, list):
            Rrs_input = np.stack(Rrs_input,2)
        
        if Rrs_input.ndim not in (2, 3) or Rrs_input.shape[-1] != len(bands):
            raise ValueError(f'Rrs input for {sensor} sensor must contain {len(bands)} components corresponding to wavelengths at: {bands}')

        init_shape = Rrs_input.shape[:-1]
        Rrs = Rrs_input.reshape(-1, Rrs_input.shape[-1]) if Rrs_input.ndim == 3 else Rrs_input
        
        
        # Load NN Chl-a models
        if self.logRrsNN:
            inputFilePath = os.path.join(os.path.dirname(__file__), 'LUTs', self.sensor,'5OWTs','NN','logRrs')
            input_1_3 = np.log10(Rrs[:, :-1])
            input_4_5 = np.log10(Rrs)
        else:
            inputFilePath = os.path.join(os.path.dirname(__file__), 'LUTs', self.sensor,'5OWTs','NN','Rrs')
            input_1_3 = Rrs[:, :-1]
            input_4_5 = Rrs
            
        scaleX_1_3, scaleY_1_3, weights_and_biases_1_3 = NN_info(os.path.join(inputFilePath,'model_clear.h5'))
        scaleX_4_5, scaleY_4_5, weights_and_biases_4_5 = NN_info(os.path.join(inputFilePath,'model_turbid.h5'))
        
        input_1_3=standardize(input_1_3, scaleX_1_3['mean'], scaleX_1_3['std'])
        input_4_5=standardize(input_4_5, scaleX_4_5['mean'], scaleX_4_5['std'])
        
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
        mask = ~np.isnan(input_1_3).any(axis=1) & ~np.isinf(input_1_3).any(axis=1)

        # Chl predictions
        Chl = np.full((Rrs.shape[0], 2), np.nan)
        if scaleY_1_3:
            Chl[mask, 0] = 10.**inverse_standardize(predict(weights_and_biases_1_3,input_1_3[mask]),
                                                    scaleY_1_3['mean'],scaleY_1_3['std'])
        else:
            Chl[mask, 0] = 10.**predict(weights_and_biases_1_3,input_1_3[mask])
        
        mask = ~np.isnan(input_4_5).any(axis=1) & ~np.isinf(input_4_5).any(axis=1)
        
        if scaleY_4_5:
            Chl[mask, 1] = 10.**inverse_standardize(predict(weights_and_biases_4_5,input_4_5[mask]),
                                                    scaleY_4_5['mean'],scaleY_4_5['std'])
        else:
            Chl[mask, 1] = 10.**predict(weights_and_biases_4_5,input_4_5[mask])
        
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

def standardize(Z, means, stds):
    """
    Standardizes the given data by subtracting the mean and dividing by the standard deviation.
    
    Parameters:
        Z (np.ndarray): The data array to standardize.
        means (np.ndarray): The mean values for each feature.
        stds (np.ndarray): The standard deviation values for each feature.
    
    Returns:
        np.ndarray: The standardized data.
    
    Example:
        standardized_data = standardize(data, mean_values, std_values)
    """
    return (Z - means) / stds

def inverse_standardize(Z, means, stds):
    """
    Reverses the standardization of data by applying the mean and standard deviations back to the data.
    
    Parameters:
        Z (np.ndarray): The standardized data array.
        means (np.ndarray): The mean values for each feature that were originally subtracted.
        stds (np.ndarray): The standard deviation values for each feature that were originally used to divide the data.
    
    Returns:
        np.ndarray: The original data before standardization.
    
    Example:
        original_data = inverse_standardize(standardized_data, mean_values, std_values)
    """

    return Z * stds + means

def predict(weights_and_biases, X):
    """
    Predicts the output using a neural network model by applying weights and biases through matrix multiplication and adding bias.
    
    Parameters:
        weights_and_biases (list of tuples): A list where each tuple contains the weights and biases for one layer of the network.
        X (np.ndarray): The input data for which predictions are needed.
    
    Returns:
        np.ndarray: The predicted output.
    
    Example:
        output = predict(weights_and_biases, input_data)
    """

    for i, (weights, biases) in enumerate(weights_and_biases):
        # Matrix multiplication and add bias
        X = np.dot(X, weights) + biases
        # Apply ReLU activation function for all but the last layer
        if i < len(weights_and_biases) - 1:  # ReLU for hidden layers
            X = relu(X)
    return X.flatten()

def relu(x):
    """
    Applies the ReLU (Rectified Linear Unit) activation function which converts all negative values in the array to zero.
    
    Parameters:
        x (np.ndarray): Data array on which to apply ReLU.
    
    Returns:
        np.ndarray: Array with ReLU applied.
    
    Example:
        activated_data = relu(data)
    """
    return np.maximum(0, x)

def NN_info(inputFilePath):
    """
    Loads neural network model information, including scale factors and weights, from a specified file path.
    
    Parameters:
        inputFilePath (str): The path to the file containing the neural network model data.
    
    Returns:
        tuple: Contains scale factors for inputs, scale factors for outputs, and a list of weights and biases for the layers.
    
    Example:
        scale_factors_x, scale_factors_y, weights_biases = NN_info('path/to/model.h5')
    """
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