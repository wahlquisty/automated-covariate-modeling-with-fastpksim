# Automated covariate modeling using efficient simulation of pharmacokinetics

This repo contains code to replicate the results and figures in the paper "Automated covariate modeling using efficient simulation of pharmacokinetics".

## Get started
Start by cloning the repo
```
git clone https://github.com/wahlquisty/automated-covariate-modeling-with-fastpksim
```

Install julia and instantiate the environment by running `julia` and in the Julia REPL, run:
```julia 
using Pkg
Pkg.instantiate(".")
```

## Main results - Covariate model from NNs

To get the main results of the paper, run in the Julia REPL:

```julia
include("NNFastPKSim.jl")
```

To plot the results and save them to csv file in folder csv/:
```
include("plotresults.jl")
```

