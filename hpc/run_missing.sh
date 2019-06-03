#!/bin/bash

# SETTINGS
datasets=("bbc" "cwebkb" "tmovie")
ksums=(5)
kleafs=(5)
layers=(2)
partitions=(2)
chains=(1)

setups=("20_missing" "40_missing" "60_missing" "80_missing")

threads=10
iterations=5000
numsamples=2500
burnin=1000

homedir="/home/"
resultsdir="/PATH/TO/results_missing/"
datadir="/PATH/TO/discrete_missing/"
codedir="/PATH/TO/BayesianSumProductNetworks"

# Actual code

# cd to code
cd ${codedir}

# ensure we use threads
export JULIA_NUM_THREADS=${threads}

# Loop over datasets and configurations...
for chain in ${chains[@]}; do
for setup in ${setups[@]}; do
for ksum in ${ksums[@]}; do
for kleaf in ${kleafs[@]}; do
for layer in ${layers[@]}; do
for partition in ${partitions[@]}; do
for dataset in ${datasets[@]}; do
    echo "Trying: Ksum ${ksum} L ${layer} Kleaf ${kleaf} J ${partition} chain ${chain} dataset ${dataset}"
    OUTDIR="${resultsdir}/${setup}/${dataset}/${ksum}_${kleaf}_${layer}_${partition}_${chain}"
    DATA="${datadir}/${setup}"
    julia scripts/run_discrete_missing_rg.jl --Ksum ${ksum} --L ${layer} --Kleaf ${kleaf} --J ${partition} --chain ${chain} --numsamples ${numsamples} ${dataset} ${DATA} ${OUTDIR} ${burnin} ${iterations}
done
done
done
done 
done
done
done
