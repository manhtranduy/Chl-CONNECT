# Chl-CONNECT
**Combination Of Neural Network models for Estimating Chlorophyll-a over Turbid and Clear Waters**

## Overview
Chl-CONNECT is a Python/MATLAB library designed for estimating chlorophyll concentrations for multiple satellite sensors including MODIS, MERIS, OLCI, and MSI. 
The library applies machine learning and statistical methods to predict chlorophyll concentrations and classify water body into different Optical Water Types.
## About
This repository contains source code for the following papers:
* Tran, M.D.; Vantrepotte, V.; Loisel, H.; Oliveira, E.N.; Tran, K.T.; Jorge, D.; Mériaux, X.; Paranhos, R. Band Ratios Combination for Estimating Chlorophyll-a from Sentinel-2 and Sentinel-3 in Coastal Waters. Remote Sens. 2023, 15, 1653. https://doi.org/10.3390/rs15061653

## Installation

### Prerequisites
Before installing and using Chl-CONNECT, please ensure your system meets the following requirements:
- Python 3.7 or later
- NumPy
- SciPy
- h5py
- Matplotlib
- Pandas
- scikit-learn
- joblib
- MATLAB (optional) R2019b or later.

Install these Python packages using pip:
```bash
pip install numpy scipy h5py tensorflow keras matplotlib pandas scikit-learn joblib
```

### Setup
Clone this repository to your local machine:
```bash
git clone https://github.com/manhtranduy/Chl-CONNECT.git
cd Chl-CONNECT
```

## Usage

### Quick Start Example
#### Python Execution
To use Chl-CONNECT in Python:
```python
from common.Chl_CONNECT import Chl_CONNECT

# Initialize the Chl_CONNECT class with specified sensor data
chl_conn = Chl_CONNECT(Rrs_input=[Rrs412, Rrs443, Rrs488, Rrs531, Rrs551, Rrs667, Rrs748], sensor='MODIS')

# Retrieve chlorophyll concentrations and Optical Water Types
Chl_modis = chl_conn.Chl_comb
Class_modis = chl_conn.Class

from common.Chl_MuBR_NDCIbased import Chl_MuBR_NDCIbased

# Initialize the Chl_MuBR_NDCIbased class with specified sensor data
chl_mubr = Chl_MuBR_NDCIbased(Rrs_input=[Rrs412, Rrs443, Rrs490, Rrs510, Rrs560, Rrs665, Rrs709], sensor='OLCI')

# Retrieve chlorophyll concentrations and Optical Water Types
Chl_olci = chl_mubr.Chl_comb
Class_olci = chl_mubr.Class

```

#### MATLAB Execution
To use compute Chl-a in MATLAB:
```matlab
addpath('./common')

[Chl_modis,Class_modis]=Chl_CONNECT({Rrs412, Rrs443, Rrs490, Rrs531, Rrs551, Rrs665, Rrs748},'sensor','modis');

[Chl_olci,Class_olci]=Chl_MuBR_NDCIbased({Rrs412, Rrs443, Rrs490, Rrs510, Rrs560, Rrs665, Rrs709},'sensor','olci');
```


### Advanced Use
For more advanced features and configurations, including adjusting parameters or using different sensors, please refer to the detailed documentation provided within the project.

## Contributing
We warmly welcome contributions to the Chl-CONNECT project. Please fork the repository, make your improvements or fixes, and submit a pull request for review.
