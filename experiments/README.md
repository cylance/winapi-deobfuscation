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

which contains the functions _run\_experiment\_1()_ and _run\_experiment\_2()_.

Alternatively, using the Makefile, run

```
make experiments 
```

Note that results vary slightly from run to run due to randomization.  For example, the HMM algorithm
optimizes a non-convex function and only finds local maximums to the likelihood function.  Also,
there is some variation due to assignment of samples to training vs. test set.  However,
these perturbations are quite minor relative to the qualitative pattern of results reported in the paper. 

### Caveat Emptor 

This research code has co-opted code originally written for other purposes, so there are classes and detours in the internal logic that are unnecessary.   Still, the code allows for reproducibility and we hope it provides a platform for further research.
