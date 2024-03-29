# massspec 
*massspec* is a Python library for the analysis of TOF mass spectra. 

It can be used to:
- convert flight times to m/z
- mark, identify, and label ion peaks
- plot multiple spectra in one figure
- calculate shot-to-shot reproducibility


Note that this module was initially developed for cryo femtosecond mass spectrometry in the Atomically Resolved Dynamics Department 
at the Max Planck Institute for the Structure and Dynamics of Matter. However, most functions within the module should be compatible with 
data acquired by other methods and/or systems.

The first few functions are based on MATLAB scripts created by Frederik Busse.

The code was written by Glaynel Alejo.

## Installation
*massspec* requires Python 3.7.3.
```
  # start by cloning the GitHub repository
  $ pip install git+https://github.com/galejo09/massspec.git
  # then install the dependencies
  $ pip install -r requirements.txt
```
