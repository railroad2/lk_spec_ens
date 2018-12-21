#!/bin/bash

#PBS -o $HOME/lk_spec_ens/log
#PBS -e $HOME/lk_spec_ens/log

if [ -n "$PBS_O_WORKDIR" ]; then
    cd $PBS_O_WORKDIR
fi

echo $PBS_O_WORKDIR

export PYTHON3=/home/kmlee/git-inst/bin/python3

pars="[0.05]"
ntest=1
lmax=11

exe="import src.test_ensemble as te; te.ens_tau(pars=$pars, ntest=$ntest, lmax_in=$lmax)"
echo $exe

$PYTHON3 -c "$exe"
