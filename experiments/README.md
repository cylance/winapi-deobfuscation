# Windows API Deobfuscation Experiments 

## Overview

This repo take the data produced by the data_collections/ folder and runs experiments. 
In particular, we create features by fitting an Hidden Markov Model (HMM) to the argument sequences from api calls, and then use that to fit a predictive model predicting the name of an api call from the argument sequences. 

The remainder of this README assumes that you have navigated to the top-level (experiments/) folder.


## Getting Started

### Installation

Run

```
python setup.py install 
```

### Setup for Development

For local development, run commands in the `Makefile` to setup a virtualenv in `env/`:

```
$ make clean-env && make env 
```

This command will perform a psuedo-installation to `env/`.  For more information, see the Makefile.

### Dependencies
See requirements.txt.  The setup commands above will load dependencies (with correct versions) automatically. 


## Usage

### Tests 

We provide unit tests so the code can be tested on small snippets of data instead of waiting for long periods of time.  Simply run:

```
nosetests -s tests/
```

Or, alternatively, using the makefile

```
make test
```


Note: Due to random assignment on a very small dataset, the unit test will rarely exit with an AssertionError. 

```
AssertionError: Must have data to run EM
```

The easy solution for now is to simply run the test again. 



### Experiments

The results and plots from the paper can be reproduced by 

```
python src/winapi_deobf_experiments/api.py
```

which contains the functions _run\_expt\_1()_ and _run\_expt\_2()_, which perform Experiments 1 and 2 in the paper. 

Alternatively, using the Makefile, run

```
make experiments 
```

Note that results can vary slightly due to randomization as the random seed changes in the configurations file.  In particular, randomization plays a role because:

1. The HMM algorithm optimizes a non-convex function and only finds local maximums to the likelihood function.   Thus, parameter initialization matters. 

2. Random assignment of samples to training vs. test set can affect results.  

However, these perturbations are quite minor relative to the qualitative pattern of results reported in the paper. 

### Caveat Emptor 

This research codebase has co-opted code originally written for other purposes, so there are classes and detours in the internal logic that are unnecessary.  Moreover, the code has not been optimized for speed or cleaned.   We hope, however, we have made it sufficiently easy to exactly reproduce results in the paper and that we have provided a point of departure for further research.