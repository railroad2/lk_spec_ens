#!/bin/bash

## fid
#qsub -l nodes=sophy06 -N likerr_lmax11_As_fid test_ens_As.sh
#qsub -l nodes=sophy04 -N rspec_lmax100_r_fid test_ens_r.sh

## nonfid
qsub -l nodes=sophy03 -N likerr_lmax11_As_nonfid test_ens_As.sh
qsub -l nodes=sophy07 -N rspec_lmax100_r_nonfid test_ens_r.sh
