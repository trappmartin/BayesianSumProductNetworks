# Bayesian Sum-Product Networks

This Julia package implements Bayesian Sum-Product Networks and infinite mixtures of Bayesian Sum-Product Networks.
Besides all necessary core routines, this package additionally implements approximate Bayesian inference using a combination of ancestral and Gibbs sampling as well as an implementation of the distributed slice sampler for infinite mixtures of SPNs.

Note that this package requires Julia version >= 1.0.

To install the package and run the provided code, make sure to first install Julia from [JuliaLang.org](https://julialang.org/downloads/).

After starting Julia, you can install the package using Julia's internal package manager.
To do so, press `]` within the Julia command line to switch to the package manager prompt. You can leaf the package manager by pressing `BACKSPACE`.

To install the package, run the following command:
```julia
pkg> add https://github.com/trappmartin/BayesianSumProductNetworks.git
```

### Dataset and Predictions
All dataset and predictions can be found under: [download](https://github.com/trappmartin/BayesianSumProductNetworks/releases/download/v1.0/data_predictions_scripts.tar)

### Running Experiments
To run the experiments, start the respective shell script located in the `hpc` folder. Those scripts are written such that they can be used as master scripts for slurm jobs.

### API
All types and functions listed here contain doc-strings. Therefore, if you are interest in more details please about the use of those functions/types please use Julia's internal documentation system. Therefore, press `?` within the Julia command line and then enter the name of the function/type you want to know more about.

The package implements Bayesian Sum-Product Networks using the following types:

```julia
RegionGraphNode{<:Real} <: AbstractRegionGraphNode
PartitionGraphNode <: AbstractRegionGraphNode
FactorizedDistributionGraphNode <: AbstractRegionGraphNode
FactorizedMixtureGraphNode <: AbstractRegionGraphNode
InfiniteSumNode
NormalInverseGamma <: Distribution
DirichletSufficientStats <: AbstractSufficientStats
NormalInvGammaSufficientStats <: AbstractSufficientStats
GammaSufficientStats <: AbstractSufficientStats
BetaSufficientStats <: AbstractSufficientStats
```

For Bayesian inference the following functions are implemented:

```julia
randomscopes!
templateRegion
templatePartition
ancestralsampling!
gibbssamplescopes!
slicesample!
logpdf
logpdf!
logmllh!
```

## License
The code is licensed under [MIT License](LICENSE).

## Citation
Please cite this work if you use it using the following [CITATION](CITATION).
