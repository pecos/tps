# This script is intended to prepare chemical rate coefficient tables
# to be ingested by tps.  The original data comes from
#
# https://github.com/pecos/lxcat-review/tree/master/bolsig/glow-discharge/nominal-rxn/Maxwellian
#
# There are two issues that must be addressed:
#   1) the original tables are too big for TPS
#   2) TPS only understands 1 table per file

import glob
import numpy as np
import h5py as h5


# The tables that Juan generates have too much data for tps (which for
# some reason wants tables with fewer than 512 entries).  So... we can
# either coarsen the resolution in temperature or restrict the
# temperature range.  Luckly Juan's original tables to up to much
# higher temperature (> 100000K) than we need in the torch.  So, I
# choose to restrict the range.

files = glob.glob("*.h5")
nmax = 500

skip_files = ["StepExcitation.h5"]

for f in files:
    print("Processing file:", f)
    h5f = h5.File(f, 'r+')

    skip = False
    for fs in skip_files:
        if (f == fs):
            skip = True

    if skip:
        print("  SKIPPPING...")
        continue

    D = h5f['table'][:,:]
    if (D.shape[0] > nmax):
        print("  New max temperature = {0:.6e}", D[nmax,0])
        Dnew = D[0:nmax,:]
        del h5f['table']
        dset = h5f.create_dataset('table', data=Dnew)
    else:
        print("  Small enough... nothing to do.")

    h5f.close()



# Some files have multiple rates.  For now this is only
# StepExcitation.h5.  Here we break them out individually.

rxns = {"E + Ar(4p) => E + Ar(m)":"4p_to_m",
        "E + Ar(4p) => E + Ar(r)":"4p_to_r",
        "E + Ar(m) => E + Ar(4p)":"m_to_4p",
        "E + Ar(m) => E + Ar(r)":"m_to_r",
        "E + Ar(r) => E + Ar(4p)":"r_to_4p",
        "E + Ar(r) => E + Ar(m)":"r_to_m"}

h5f = h5.File("StepExcitation.h5", "r+")
for key, val in rxns.items():
    print("Working on ", key, "...")
    D = h5f[key][:,:]
    Dm = D[0:nmax,:]

    h5fnew = h5.File("StepExcitation_"+val+".h5", "w")
    dset = h5fnew.create_dataset('table', data=Dm)
    h5fnew.close()

h5f.close()
