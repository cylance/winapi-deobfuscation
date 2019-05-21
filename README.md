# Towards Generic Deobfuscation of Windows API Calls

## Paper 

### Official version 

The official version of the paper can be found [here](https://www.ndss-symposium.org/wp-content/uploads/2018/07/bar2018_11_Kotov_paper.pdf). 

### Citation 
Kotov, V., & Wojnowicz, M. (2018). Towards Generic Deobfuscation of Windows API Calls. In _Proceedings of Workshop on Binary Analysis Research (BAR 2018)_ (pp. 1-11). Reston, VA: Internet Society. https://dx.doi.org/10.14722/bar.2018.23011

## Code 

We provide source code to replicate the data collection process and experimental results.


### data_collection
This folder contains the simplified symbolic execution engine and scripts to extract API call information from 32-bit Windows executables; as well as prepare the data to be fed into our HMM-based classifier.

### experiments
This folder has all the code required to replicate both experiments described in the paper. It takes in the data prepared using the scripts from data_collection folder.


