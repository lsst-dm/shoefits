# -*- python -*-
import os

from lsst.sconsUtils import scripts, state

scripts.BasicSConstruct("shoefits", disableCc=True)
mypy = state.env.Command("mypy.log", "python/lsst/shoefits", "mypy python/lsst 2>&1 | tee -a mypy.log")
state.env.Alias("mypy", mypy)

# Propagate environment variables used only by this package through SCons.
envvars = ["LSST_SHOEFITS_TEST_TMP"]
for e in envvars:
    if e in os.environ:
        state.env["ENV"][e] = os.environ[e]
