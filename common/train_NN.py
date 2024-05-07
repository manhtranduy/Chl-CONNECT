from sklearn.model_selection import train_test_split
from common.utils import ErrorNorm as ern
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class train_NN:
    def __init__(self, 
                 X: np.ndarray,
                 Y: np.ndarray,
                 noiseLevel: np.ndarray = [0.518422,0.371655,0.284737,0.216589,0.201913,0.430994,0.715927],
                 maxNeurons: int = 128,
                 logTransformX: bool = False,
                 logTransformY: bool = True,
                 maxLayer: int = 3,
                 divisionFactor: int = 2,
                 numberOfIters: int = 1,
                 scalerX_actv: bool = True,
                 scalerY_actv: bool = False,
                 solver: str = 'adam',
                 activationFunction: str ='relu',
                 scoreThreshold: float = 0.72,
                 testSize: float = 0.3,
                 l2RegularizationRange: tuple = (0, 0.11, 0.01),
                 scaler:str = 'standard' # 'min_max' or 'standard'
                 ):
        # Initialize class attributes here
        self.X = X
        self.Y = Y
        self.noiseLevel = noiseLevel
        self.maxNeurons = maxNeurons
        self.divisionFactor = divisionFactor
        self.numberOfIters = numberOfIters
        # self.solver = solver
        self.activationFunction = activationFunction
        self.scoreThreshold = scoreThreshold
        self.l2RegularizationRange = l2RegularizationRange
        self.maxLayer = maxLayer
        self.logTransformX = logTransformX
        self.logTransformY = logTransformY
        self.solver = solver
        
        
        
        if not self.X.shape[0] == self.Y.shape[0]:
            raise ValueError("X and Y must have the same number of samples.")
        
        # Get valid indices
        # self.valid_indices = ~np.isnan(self.X).any(axis=1) & ~np.isnan(self.Y) & \
        #                      ~np.any(self.X <= 0, axis=1) & ~(self.Y<=0)
        self.valid_indices = ~np.isnan(self.X).any(axis=1) & ~np.isnan(self.Y) & \
                              ~np.any(self.X == 0, axis=1)

        
        self.X = self.X[self.valid_indices,:]
        self.Y = self.Y[self.valid_indices]

        total_indices = np.arange(len(self.Y))
        self.train_indices, self.val_indices = train_test_split(total_indices, test_size=testSize)
        X_init=self.X
        
        # Log-transformed
        if self.logTransformX:
            # self.Y=np.log10(self.Y)
            self.X=np.log10(self.X)
            # Generate noise
            self.X_noisy=np.log10(add_noise(X_init,noise_level=self.noiseLevel))
        else:
            self.X_noisy=add_noise(X_init,noise_level=self.noiseLevel)
            
        if self.logTransformY:
            self.Y = np.log10(self.Y)

        # Prepare data according to valid indices
        self.X_val_noisy=self.X_noisy[self.val_indices,:]
        
        self.X_train=self.X[self.train_indices,:]
        self.X_val=self.X[self.val_indices,:]
        self.Y_train=self.Y[self.train_indices]
        self.Y_val=self.Y[self.val_indices]
        
        # self.X_val_noisy=add_noise(self.X_val,noise_level=self.noiseLevel)
        # Standardize data
        # Get mean and standard deviation
        # self.scaler = StandardScaler()
        
        if scalerX_actv:
            if scaler == 'min_max':
                self.scalerX = MinMaxScaler()
            else:
                self.scalerX = StandardScaler()
                    
            self.X_train = self.scalerX.fit_transform(self.X_train)
            self.X_val = self.scalerX.transform(self.X_val)
            self.X_val_noisy = self.scalerX.transform(self.X_val_noisy)
        
        if scalerY_actv:
            if scaler == 'min_max':
                self.scalerY = MinMaxScaler()
            else:
                self.scalerY = StandardScaler()
            self.Y_train = self.scalerY.fit_transform(self.Y_train.reshape(-1, 1)).ravel()
            self.Y_val = self.scalerY.transform(self.Y_val.reshape(-1, 1)).ravel()

        

        self.best_score_train=-np.inf
        self.best_score_val=-np.inf
        self.best_score_val_noisy=-np.inf
        self.best_structure=None
        
        self.model_list=[]
        self.score_val_list=[]
        self.score_train_list=[]
        self.score_val_noisy_list=[]
        self.arc_list=[]
    
        for layer in range(2, self.maxLayer + 1):
            for l2_reg_val in np.arange(self.l2RegularizationRange[0], self.l2RegularizationRange[1], self.l2RegularizationRange[2]):
                    model, score_train, score_val, score_val_noisy,arc=train_model(self.X_train,self.Y_train,
                                                                                   self.X_val,self.Y_val,self.X_val_noisy,
                                                                    n_layer=layer,
                                                                    n_iter=self.numberOfIters,
                                                                    max_neurons=self.maxNeurons,
                                                                    n_factor=self.divisionFactor,
                                                                    solver=self.solver,
                                                                    l2_reg=l2_reg_val)
                    self.model_list.append(model)
                    self.score_val_list.append(score_val)
                    self.score_train_list.append(score_train)
                    self.score_val_noisy_list.append(score_val_noisy)
                    self.arc_list.append(arc)

        self.scoreThreshold_val = np.median(self.score_val_list)
        self.scoreThreshold_noisy = np.median(self.score_val_noisy_list)
                
        good_ind= (self.score_val_list > self.scoreThreshold_val) & (self.score_val_noisy_list > self.scoreThreshold_noisy)
        max_val = np.max(np.array(self.score_val_list)[good_ind])
        ind = self.score_val_list == max_val
        ind=int(np.where(ind)[0])
        # if score_val_noisy > best_score_val_noisy:
        self.best_score_train = self.score_train_list[ind]
        self.best_score_val = self.score_val_list[ind]
        self.best_score_val_noisy = self.score_val_noisy_list[ind]
        self.best_model = self.model_list[ind]
        self.best_structure = self.arc_list[ind]
    
        print(f"The overall best network structure is: {self.best_structure} with R^2 val = {self.best_score_val} \
              and R^2 val (noisy) = {self.best_score_val_noisy}")
              
def standardizeData(X, mean_vals, std_vals):
    """
    Scale the dataset X manually using provided mean and standard deviation values.
    
    Parameters:
    - X: numpy array, dataset to scale.
    - mean_vals: numpy array, mean values for each feature.
    - std_vals: numpy array, standard deviation for each feature.
    
    Returns:
    - Scaled dataset.
    """
    return (X - mean_vals) / std_vals

        

def add_noise(data, noise_level=0.5):
    
    if not isinstance(noise_level, list):
        noise_level=[noise_level]
    if len(noise_level)<data.shape[1]:
        for i in range(data.shape[1]):
            noise_level.append(noise_level[0])
            
    noisy_data = np.copy(data)
    for i in range(data.shape[1]):
        # print(i)
        # noise = abs(np.random.normal(0, noise_level[i],data.shape[0]))
        noise = np.random.normal(0, noise_level[i],data.shape[0])
        noisy_data[:, i] =noisy_data[:, i]+ noise*noisy_data[:, i]
    return noisy_data



def create_model(n_layers,
                 max_neurons=150,
                 n_factor=5,
                 solver='adam',
                 activation='relu',
                 l2_reg=0.15):
    arc=()
    current_neurons = max_neurons
    for n_layer in range(n_layers):
        arc=arc+(current_neurons,)
        current_neurons=int(current_neurons/n_factor)
    
    # print(solver)
    model = MLPRegressor(
        hidden_layer_sizes=arc,
        activation=activation,
        solver=solver,
        alpha=l2_reg,  # Slightly increased
        learning_rate='adaptive',
        learning_rate_init=0.001,
        batch_size=32,
        max_iter=512,  # Increased
        shuffle=False,
        random_state=42,  # Set for reproducibility
        verbose=False,
        warm_start=True,
        nesterovs_momentum=True,
        early_stopping=True,  # Enabled
        validation_fraction=0.3,
        epsilon=1e-08,
        n_iter_no_change=4,
        tol=1e-4,
    )
    return model,arc


def train_model(X_train,Y_train,X_val,Y_val,X_val_noisy,
                n_layer=3,
                n_iter=5,
                max_neurons=100,
                n_factor=2,
                l2_reg=0.15,
                solver='adam'):
    valid_ind=~(np.any(np.isinf(X_train), axis=1)) & ~(np.isinf(Y_train))
    X_train=X_train[valid_ind,:]
    Y_train=Y_train[valid_ind]
    
    valid_ind=~(np.any(np.isinf(X_val), axis=1)) & ~(np.isinf(Y_val))
    X_val=X_val[valid_ind,:]
    Y_val=Y_val[valid_ind]
    X_val_noisy=X_val_noisy[valid_ind,:]
    
        # np.where(np.any(X_train==0,axis=1))
        # np.where(np.any(np.isinf(X_train),axis=1))
        # np.where(np.any(np.isnan(X_train),axis=1))
        
        # np.where(np.any(X_val_noisy==0))
        # np.where(np.any(np.isinf(X_val_noisy)))
        # np.where(np.any(np.isnan(X_val_noisy),axis=1))
        
        # np.where(np.any(self.X_val==0,axis=1))
        # np.where(np.any(np.isinf(self.X_val),axis=1))
        # np.where(np.any(np.isnan(self.X_val),axis=1))
        
        # np.where(np.any(self.Y_val==0))
        # np.where(np.any(np.isinf(self.Y_val)))
        # np.where(np.any(np.isnan(self.Y_val)))
        
    
    best_model = None
    best_score_train = -np.inf
    best_score_val = -np.inf
    best_score_val_noisy = -np.inf
    best_arc = None
    for attempt in range(n_iter):  # Train 3 times for each architecture
        model, arc = create_model(n_layer, 
                                  max_neurons=max_neurons, 
                                  n_factor=n_factor, 
                                  l2_reg=l2_reg,
                                  solver=solver)
        model.fit(X_train, Y_train)
        
        # Evaluate on original training data
        train_pred = model.predict(X_train).flatten()
        # score_train = r2_score(Y_train, train_pred)
        score_train = (ern(Y_train, train_pred).Slope[0] + r2_score(Y_train, train_pred))/2
        
        # Evaluate on original validation data
        val_pred = model.predict(X_val).flatten()
        # score_val = r2_score(Y_val, val_pred)
        score_val = (ern(Y_val, val_pred).Slope[0] + r2_score(Y_val, val_pred))/2
        
        # X_val_noisy=add_noise(X_val)
        # Evaluate on noise-augmented validation data
        mask=~np.any(np.isnan(X_val_noisy),axis=1)
        X_val_noisy=X_val_noisy[mask,:]
        val_pred_noisy = model.predict(X_val_noisy).flatten()
        # score_val_noisy = r2_score(Y_val[mask], val_pred_noisy)
        score_val_noisy = (ern(Y_val, val_pred_noisy).Slope[0] + r2_score(Y_val, val_pred_noisy))/2
        # val_pred_noisy = model.predict(X_val_noisy).flatten()
        # score_val_noisy = r2_score(Y_val, val_pred_noisy)
    
        # Update the best model if improvement is seen in both datasets
        # if score_train > best_score_train and score_val > best_score_val and score_val_noisy > best_score_val_noisy:
        if score_val > best_score_val and score_val_noisy > best_score_val_noisy:
            best_score_train=score_train
            best_score_val = score_val
            best_score_val_noisy = score_val_noisy
            best_model = model
            best_arc = arc
    # After finding the best model for the current architecture
    if best_model:
        print(f"Best model for architecture {arc} with R^2 val = {best_score_val} and R^2 val (noisy) = {best_score_val_noisy}")

    return best_model, best_score_train, best_score_val, best_score_val_noisy, best_arc