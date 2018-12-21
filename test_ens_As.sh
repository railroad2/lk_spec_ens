#!/bin/bash

#PBS -o $HOME/lk_spec_ens/log
#PBS -e $HOME/lk_spec_ens/log

if [ -n "$PBS_O_WORKDIR" ]; then
    cd $PBS_O_WORKDIR
fi

echo $PBS_O_WORKDIR

export PYTHON3=/home/kmlee/git-inst/bin/python3

#pars="[1.0e-9, 4e-9]"
vals=`seq -s ', ' 1e-9 0.5e-9 4e-9`
pars="[$vals]"
ntest=1000
lmax=11
specplt="False"
nonfid="True"

exe="import src.test_ensemble as te; te.ens_As(pars=$pars, ntest=$ntest, lmax_in=$lmax, specplt=$specplt, nonfid=$nonfid)"
echo $exe

$PYTHON3 -c "$exe"
