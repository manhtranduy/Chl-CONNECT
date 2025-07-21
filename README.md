# Chl-CONNECT
**Combination Of Neural Network models for Estimating Chlorophyll-a over Turbid and Clear Waters**

## Overview
Chl-CONNECT is a Python/MATLAB library designed for estimating Chlorophyll-a concentrations (Chl-a) for multiple satellite sensors including MODIS, MERIS, OLCI, and MSI. 
The library applies machine learning and statistical methods to predict chlorophyll concentrations and classify water body into different Optical Water Types.

### Key Features
- 🛰️ Multi-sensor support: MODIS, MERIS, Sentinel-3/OLCI, Sentinel-2/MSI
- 🤖 Chlorophyll-a estimation
- 🌊 Optical Water Type (OWT) classification


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

### Performance Metrics
Our model has been validated against an extensive in-situ/matchup dataset gathered over various environments associated with different trophic levels.

## In-situ Validation Results
<p align="center">
  <img src="docs/images/DST_MODIS_benchmark.png" alt="Chlorophyll-a estimation performance on the Test in-situ dataset" width="800">
</p>

| Class   | R²<sub>log</sub> | Slope<sub>log</sub> | MAPD (%)  | SSPB (%)   | N   | Not Valid | area<sub>norm</sub> | Model     |
|:--------|:----------------:|:------------------:|:----------:|:-----------:|:----|:----------|:--------------------:|:----------|
| OWT 1   | **0.526** | 0.700              | 41.183     | **4.050** | 95  | 0         | **0.640** | CONNECT   |
| OWT 1   | 0.457            | **0.769** | 49.301     | -72.214    | 95  | 0         | 1.248                | MuBR      |
| OWT 1   | 0.466            | 0.653              | **33.139** | 12.504     | 95  | 0         | 0.767                | OC3M      |
| OWT 1   | 0.380            | 0.697              | 36.376     | -8.952     | 91  | 4         | 1.228                | OC5-Gohin |
| OWT 2   | 0.573            | **0.753** | 43.434     | **3.344** | 50  | 0         | **0.386** | CONNECT   |
| OWT 2   | **0.660** | 0.564              | 37.341     | -50.905    | 50  | 0         | 0.601                | MuBR      |
| OWT 2   | 0.538            | 0.249              | **36.973** | -3.842     | 50  | 0         | 0.603                | OC3M      |
| OWT 2   | 0.282            | 0.291              | 38.501     | -21.872    | 50  | 0         | 1.064                | OC5-Gohin |
| OWT 3   | **0.351** | 0.669              | **37.569** | **1.487** | 283 | 0         | **0.612** | CONNECT   |
| OWT 3   | 0.196            | **0.825** | 44.990     | -18.499    | 283 | 0         | 0.712                | MuBR      |
| OWT 3   | 0.152            | 0.655              | 55.836     | 36.955     | 283 | 0         | 1.427                | OC3M      |
| OWT 3   | 0.197            | 0.743              | 46.237     | 9.863      | 283 | 0         | 0.803                | OC5-Gohin |
| OWT 4   | **0.702** | **0.870** | **33.802** | **-3.737** | 344 | 0         | **0.148** | CONNECT   |
| OWT 4   | 0.515            | 0.597              | 51.498     | -35.330    | 344 | 0         | 0.745                | MuBR      |
| OWT 4   | 0.152            | 0.572              | 63.927     | -58.791    | 344 | 0         | 1.427                | OC3M      |
| OWT 4   | 0.208            | 0.788              | 61.950     | -49.994    | 340 | 4         | 1.717                | OC5-Gohin |
| OWT 5   | **0.643** | 0.664              | **42.244** | **21.630** | 32  | 0         | **0.198** | CONNECT   |
| OWT 5   | 0.012            | 0.562              | 230.281    | 226.145    | 32  | 0         | 1.427                | MuBR      |
| OWT 5   | 0.416            | 0.615              | 100.211    | 100.022    | 32  | 0         | 0.554                | OC3M      |
| OWT 5   | 0.208            | **0.987** | 82.071     | 64.594     | 31  | 1         | 0.457                | OC5-Gohin |
| Overall | **0.870** | 0.961              | **36.769** | **1.168** | 804 | 0         | **0.173** | CONNECT   |
| Overall | 0.777            | **0.963** | 49.477     | -31.877    | 804 | 0         | 0.548                | MuBR      |
| Overall | 0.699            | 0.780              | 56.138     | -3.469     | 804 | 0         | 1.000                | OC3M      |
| Overall | 0.698            | 0.857              | 51.441     | -15.322    | 795 | 9         | 1.556                | OC5-Gohin |


### Advanced Use
For more advanced features and configurations, including adjusting parameters or using different sensors, please refer to the detailed documentation provided within the project.

## Contributing
We warmly welcome contributions to the Chl-CONNECT project. Please fork the repository, make your improvements or fixes, and submit a pull request for review.
