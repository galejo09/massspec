# massspec 
*massspec* is a Python library for the analysis of mass spectra. 

It can be used to:
- convert flight times to m/z
- mark, identify, and label ion peaks
- plot multiple spectra in one figure
- calculate shot-to-shot reproducibility


Note that this module was developed specifically for cryo femtosecond mass spectrometry in the Atomically Resolved Dynamics Department 
at the Max Planck Institute for the Structure and Dynamics of Matter. Therefore not all the functions within the module may be compatible with 
data acquired by other methods and/or systems.

## Installation
massspec requires Python 3.7.3.
```
  # start by cloning the GitHub repository
  $ pip install git+https://github.com/galejo09/massspec.git
  # then install the dependencies
  $ pip install -r requirements.txt
```
