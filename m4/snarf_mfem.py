#!/usr/bin/env python3
import argparse
import os
import sys
import logging

logging.basicConfig(format="%(message)s",level=logging.INFO,stream=sys.stdout)

parser = argparse.ArgumentParser()
parser.add_argument("--dir",type=str,help="MFEM directory to scan")
parser.add_argument("--libs",     dest='parseLibs',action='store_true',help="Parse external libraries")
parser.add_argument("--includes", dest='parseIncs',action='store_true',help="Parse include paths")
parser.add_argument("--noreplace",dest='noReplace',action='store_true',help="Do not replace paths with env variables")
args = parser.parse_args()

if args.parseLibs and args.parseIncs:
    logging.error("[ERROR]: Options --libs and --includes cannot be run simultaneously.")
    exit(1)

if not args.parseLibs and not args.parseIncs:
    logging.error("[ERROR]: Please run with either the --libs or --includes option defined.")
    exit(1)    

if args.dir:
    mfem_dir = args.dir
    print("dir = %s\n" % args.dir)
else:
    mfem_dir = os.getenv("MFEM_DIR")
    if not mfem_dir:
        logging.error("Set MFEM_DIR or use --dir to provide MFEM install path.")
        exit(1)
if not os.path.isdir(mfem_dir):
    logging.error("Unable to access MFEM installation path: %s" % mfem_dir)


configFile = mfem_dir + "/share/mfem/config.mk"    
if not os.path.exists(configFile):
    logging.error("Unable to access config file: %s" % configFile)
    exit(1)
else:
    logging.debug("Parsing config file: %s" % configFile)

def parse_includes(incs,match):
    known_libs = ["hypre","metis","netcdf","petsc","superlu_dist"]

    myincs = incs.lstrip(match)    # strip leading variable name (match)
    myincs = myincs.lstrip()       # strip spaces after variable name
    myincs = myincs.lstrip("=")    # strip equal sign
    myincs = myincs.lstrip()       # strip spaces after equal sign

    incs = []
    for item in myincs.split():
        # check for seach paths
        if item.startswith("-I"):
            logging.debug("-I %s (include searchpath option detected)" % item)
            # check if known 3rd party lib, replace path with module equivalent if available
            found = False
            for ext_lib in known_libs:
                if "/" + ext_lib + "/" in item:
                    logging.debug("known 3rd party lib found -> %s" % ext_lib)
                    moduleDir = os.getenv(ext_lib.upper()+"_INC")
                    if moduleDir and not args.noReplace:
                        if os.path.isdir(moduleDir):
                            incdir=item.strip("-I")
                            # verify the path matches the env variable
                            if incdir == moduleDir:
                                found = True
                                searchDir="$" + '{' + ext_lib.upper() + "_INC" + '}'
                                logging.debug("Replacing header search path with %s" % searchDir)
                                incs.append("-I" + searchDir)
                                break
            if not found:
                incs.append(item)
        else:
            incs.append(item)

    return(' '.join(incs))

def parse_libs(libs,match):
    known_libs = ["hypre","metis","netcdf","petsc","superlu_dist"]
    
    mylibs = libs.lstrip(match)    # strip leading variable name (match)
    mylibs = mylibs.lstrip()       # strip spaces after variable name
    mylibs = mylibs.lstrip("=")    # strip equal sign
    mylibs = mylibs.lstrip()       # strip spaces after equal sign

    libs = []
    for item in mylibs.split():
        # check for seach paths
        if item.startswith("-L"):
            logging.debug("-L %s (searchpath option detected)" % item)
            # check if known 3rd party lib, replace path with module equivalent if available
            found = False
            for ext_lib in known_libs:
                if "/" + ext_lib + "/" in item:
                    logging.debug("known 3rd party lib found -> %s" % ext_lib)
                    moduleDir = os.getenv(ext_lib.upper()+"_LIB")
                    if moduleDir and not args.noReplace:                    
                        if os.path.isdir(moduleDir):
                            libdir=item.strip("-L")
                            # verify the path matches the env variable
                            if libdir == moduleDir:
                                logging.debug("have a match")
                                found = True
                                searchDir="$" + "{" + ext_lib.upper() + "_LIB" + "}"
                                logging.debug("Replacing library search path with %s" % searchDir)
                                libs.append("-L" + searchDir)
                                break
            if not found:
                libs.append(item)
        else:
            libs.append(item)

    return(' '.join(libs))

with open(configFile,"r") as infile:
    for line in infile:
        if args.parseLibs:
            if line.startswith("MFEM_EXT_LIBS"):
                libs = parse_libs(line.strip(),"MFEM_EXT_LIBS")
                print(libs)
        if args.parseIncs:
            if line.startswith("MFEM_TPLFLAGS"):
                incs = parse_includes(line.strip(),"MFEM_TPLFLAGS")
                print(incs)
            
