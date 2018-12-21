#!/bin/bash

#PBS -o $HOME/lk_spec_ens/log
#PBS -e $HOME/lk_spec_ens/log

if [ -n "$PBS_O_WORKDIR" ]; then
    cd $PBS_O_WORKDIR
fi

echo $PBS_O_WORKDIR

export PYTHON3=/home/kmlee/git-inst/bin/python3

pars="[0.025, 0.03, 0.050]"
ntest=1000
lmax=100
specplt="True"

exe="import src.test_ensemble as te; te.ens_r(pars=$pars, ntest=$ntest, lmax_in=$lmax, specplt=$specplt)"
echo $exe

$PYTHON3 -c "$exe"
