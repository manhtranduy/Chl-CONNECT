# Chl-CONNECT
**Combination Of Neural Network models for Estimating Chlorophyll-a over Turbid and Clear Waters**

## Overview
Chl-CONNECT is a comprehensive Python library designed for analyzing and classifying chlorophyll concentrations using satellite sensor data. The library applies advanced machine learning and statistical methods to predict and classify chlorophyll data accurately, making it invaluable for environmental and ecological research.

## Installation

### Prerequisites
Before installing and using Chl-CONNECT, please ensure your system meets the following requirements:
- Python 3.7 or later: Essential for running the Python scripts.
- NumPy: For handling large arrays and matrices.
- SciPy: Used for scientific computing and technical computing.
- h5py: Necessary for handling H5 model files in Python.
- TensorFlow or Keras: For building and running neural network models.
- Matplotlib: Optional but recommended for visualizing data.
- Pandas: For data manipulation and analysis, particularly used in classification_functions.py.
- scikit-learn: May be required for any machine learning models or additional statistical functions used in classification_functions.py.
- joblib: For efficient model saving and loading.
- MATLAB (optional): Required only if you are running MATLAB scripts or translating MATLAB code to Python.

Install these Python packages using pip:
```bash
pip install numpy scipy h5py tensorflow keras matplotlib pandas scikit-learn joblib

### Setup
Clone this repository to your local machine:

git clone https://github.com/manhtranduy/Chl-CONNECT.git
cd Chl-CONNECT

## Usage

### Quick Start Example
To use Chl-CONNECT for analyzing chlorophyll data in Python:
from Chl_CONNECT import Chl_CONNECT

Assuming Rrs data arrays are defined as NumPy arrays
Rrs412, Rrs443, Rrs488, Rrs531, Rrs551, Rrs667, Rrs748 = [your_rrs_data_here]

Initialize the Chl_CONNECT class with specified sensor data
chl_conn = Chl_CONNECT(Rrs_input=[Rrs412, Rrs443, Rrs488, Rrs531, Rrs551, Rrs667, Rrs748], sensor='MODIS')

Retrieve chlorophyll concentrations and classification results
Chl = chl_conn.Chl
Class = chl_conn.Class

print("Chlorophyll Concentrations:", Chl)
print("Classification Results:", Class)

### Advanced Use
For more advanced features and configurations, including adjusting parameters or using different sensors, please refer to the detailed documentation provided within the project or in the docs directory.

## Contributing
We warmly welcome contributions to the Chl-CONNECT project. Please fork the repository, make your improvements or fixes, and submit a pull request for review.

## Support
For support, issue tracking, and feature requests, please use the GitHub issues section of this repository.

## License
Chl-CONNECT is licensed under the MIT License - see the LICENSE file for more details.
