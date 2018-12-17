#!/bin/bash

#PBS -o $HOME/lk_spec_ens/log
#PBS -e $HOME/lk_spec_ens/log

if [ -n "$PBS_O_WORKDIR" ]; then
    cd $PBS_O_WORKDIR
fi

echo $PBS_O_WORKDIR

export PYTHON3=/home/kmlee/git-inst/bin/python3

pars="[2.0e-9]"
ntest=1
lmax=11

exe="import test_ensemble; test_ensemble.ens_As(pars=$pars, ntest=$ntest, lmax_in=$lmax)"
echo $exe

$PYTHON3 -c "$exe"
