from .tabulated_reaction_rates import TabulatedSolver
from .utils import master_print

# set path to C++ TPS library

import pathlib
import sys

path = pathlib.Path(__file__).parent.resolve()
sys.path.append(path + "/.libs")
from libtps import TPS