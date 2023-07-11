#!/usr/bin/env python3
import argparse
import sys
import logging
import mpi4py
import mpi4py.MPI as MPI

logging.basicConfig(format="%(message)s",level=logging.INFO,stream=sys.stdout)

parser = argparse.ArgumentParser()
parser.add_argument("--lib",     dest='get_lib',action='store_true',help="Get library")
parser.add_argument("--include", dest='get_inc',action='store_true',help="Get include path")
args = parser.parse_args()

if args.get_lib and args.get_inc:
    logging.error("[ERROR]: Options --lib and --inc cannot be run simultaneously.")
    exit(1)

if args.get_lib:
    print(MPI.__file__)

if args.get_inc:
    print(mpi4py.get_include())
