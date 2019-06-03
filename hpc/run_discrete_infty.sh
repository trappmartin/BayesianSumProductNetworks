#!/bin/bash

# SETTINGS
datasets=("accidents" "ad" "baudio" "bbc" "bnetflix" "book" "c20ng" "cr52" "cwebkb" "dna" "jester" "kdd" "kosarek" "msnbc" "msweb" "nltcs" "plants" "pumsb_star" "tmovie" "tretail")
ksums=(5 10)
kleafs=(5 10)
layers=(2 4)
partitions=(2 4 8)
chains=(1)

threads=10
iterations=10000
numsamples=10000
burnin=5000

homedir="/home/"
resultsdir="/PATH/TO/results_infty"
datadir="/PATH/TO/discrete"
codedir="/PATH/TO/BayesianSumProductNetworks"

# Actual code

# cd to code
cd ${codedir}

# ensure we use threads
export JULIA_NUM_THREADS=${threads}

# Loop over datasets and configurations...
for chain in ${chains[@]}; do
for ksum in ${ksums[@]}; do
for kleaf in ${kleafs[@]}; do
for layer in ${layers[@]}; do
for partition in ${partitions[@]}; do
for dataset in ${datasets[@]}; do
    echo "Trying (infty): Ksum ${ksum} L ${layer} Kleaf ${kleaf} J ${partition} chain ${chain} dataset ${dataset}"
    if [ "${kleaf}" -ge "${ksum}" ]; then
        OUTDIR="${resultsdir}/${dataset}/${ksum}_${kleaf}_${layer}_${partition}_${chain}"
        julia scripts/run_discrete_rg_infty.jl --Ksum ${ksum} --L ${layer} --Kleaf ${kleaf} --J ${partition} --chain ${chain} --numsamples ${numsamples} ${dataset} ${datadir} ${OUTDIR} ${burnin} ${iterations}
    else
        echo "skipping"
    fi
done
done
done
done
done
done
