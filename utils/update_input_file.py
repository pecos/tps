#!/usr/bin/env python3
import argparse
import os
import sys
import logging
import numpy
import configparser

parser = argparse.ArgumentParser()
parser.add_argument("--infile",type=str,help="existing input file to convert",required=True)
args = parser.parse_args()

inFile = args.infile

isDryRun = False

if not os.path.exists(inFile):
    logging.error("Unable to access file: %s" % inFile)
    exit(1)

# setup mapping from old inputs vars to new keyword/sections
flow = [ ["MESH","mesh"],
         ["POL_ORDER","order"],
         ["INT_RULE","integrationRule"],
         ["BASIS_TYPE","basisType"],
         ["EQ_SYSTEM","equations"],
         ["REF_LENGTH","refLength"],
         ["USE_ROE","useRoe"],
         ["ITERS_OUT","outputFreq"],
         ["NMAX","maxIters"],
         ["IS_SBP","enableSummationByParts"],
         ["FLUID","fluid"],
         ["BULK_VISC_MULT","bulkViscosityMultiplier"],
         ["VISC_MULT","viscosityMultiplier"],
         ["GRAD_PRESSURE","additionalGradPressure"],
       ]

passiveScalars = [ ["PASIVE_SCALAR",""] ]

viscSource = [ ["LV_PLANE_NORM","norm"],
               ["LV_PLANE_P0","p0"],
               ["LV_PLANE_PINIT","pInit"],
               ["LV_VISC_RATIO","viscosityRatio"]
]

time = [ ["CFL","cfl"],
         ["TIME_INTEGRATOR","integrator"],
         ["DT_CONSTANT","enableConstantTimestep"],
       ]
ics  = [ ["INIT_RHO","rho"],
         ["INIT_RHOVX","rhoU"],
         ["INIT_RHOVY","rhoV"],
         ["INIT_RHOVZ","rhoW"],
         ["INIT_P","pressure"] ]

sponge = [ ["SZ_PLANE_NORM",""],
           ["SZ_PLANE_P0",""],
           ["SZ_PLANE_PINIT",""],
           ["SZ_TYPE","type"],
           ["SZ_MULT","multiplier"]           
]

rms = [ ["ENABLE_AUTORESTART","enableAutoRestart"],
        ["RM_THRESHOLD","timeThreshold"],
        ["RM_CHECK_FREQUENCY","checkFreq"],
]
           

io   = [ ["OUTPUT_NAME","outdirBase"],
         ["RESTART_CYCLE","enableRestart"],
         ["RESTART_FROM_AUX","restartMode"],
         ["RESTART_SERIAL","restartMode"],
       ]

averaging = [ ["CALC_MEAN_RMS","enableAveraging"],
              ["CONTINUE_MEAN_CALC","enableContinuation"],
              ["SAVE_MEAN_HIST","saveMeanHist"]
            ]

sections = {}
sections["flow"] = flow
sections["time"] = time
sections["initialConditions"] = ics
sections["io"]  = io
sections["averaging"] = averaging
sections["spongezone"] = sponge
sections["jobManagement"] = rms
sections["viscousSource"] = viscSource
sections["passiveScalars"] = passiveScalars

newFile = configparser.ConfigParser()
newFile.optionxform = str
newFile.add_section("solver")
newFile["solver"]["type"] = "flow"
    
numInlets     = 0
numOutlets    = 0
numWalls      = 0
numScalars    = 0
inletMapping  = {'0':'subsonic','1':'nonReflecting','2':'nonReflectingConstEntropy'}
outletMapping = {'0':'subsonicPressure','1':'nonReflectingPressure',
                 '2':'nonReflectingMassFlow','3':'nonReflectingPointBasedMassFlow'}
wallMapping   = {'0':'inviscid','1':'viscous_adiabatic','2':'viscous_isothermal'}

# map old name to new section/name
def getNewName(varName):
    found = False
    for key,value in sections.items():
        origNames = numpy.array(value)
        if varName in origNames[:,0]:
            found = True
            index = numpy.where(origNames[:,0] == varName)[0][0]
            newName = value[index][1]
            return(key,newName)
    if not found:
        logging.error("No section defined for var -> %s" % varName)
        exit(1)

# add new section if not present already
def addSection(section):
    if not newFile.has_section(section):
        newFile.add_section(section)

# deal with special inputs that are multi-valued on a case-by-case basis        
def handleMultiValuedInputs(entry):

    global numInlets, numOutlets, numWalls, numScalars
    entries = entry.split()
    varName = entries[0]
    if varName == "WALL":

        patch = entries[1]
        numWalls += 1
        section = "boundaryConditions/wall" + str(numWalls)        

        addSection(section)
        newFile[section]["patch"] = patch
        newFile[section]["type"]  = wallMapping[entries[2]]
        if wallMapping[entries[2]] == 'viscous_isothermal':
            newFile[section]["temperature"] = entries[3]
        elif wallMapping[entries[2]] == 'inviscid':
            # no additional vars required
            noop=True
        else:
            logging.error("Unsupported wall BC type -> %s" % wallMapping[entries[2]])
            exit(1)
    elif varName == "PASSIVE_SCALAR":
        numScalars += 1
        section = "passiveScalar" + str(numScalars)
        addSection(section)
        xyz = str(entries[1]) + " " + str(entries[2]) + " " + str(entries[3])
        newFile[section]["xyz"]    = "'" + xyz + "'"
        newFile[section]["radius"] = entries[4]
        newFile[section]["value"]  = entries[5]
    elif varName == "INLET":

        patch = entries[1]
        numInlets += 1
        section = "boundaryConditions/inlet" + str(numInlets)        

        addSection(section)
        newFile[section]["patch"] = patch        
        newFile[section]["type"]  = inletMapping[entries[2]]
        if inletMapping[entries[2]] == 'subsonic':
            newFile[section]["density"] = entries[3]
            uvw = "'" + str(entries[4]) + " " + str(entries[5]) + " " + str(entries[6]) + "'"
            newFile[section]["uvw"] = uvw
        else:
            logging.error("Unsupported inlet BC type -> %s" % inletMapping[entries[2]])
            exit(1)
    elif varName == "OUTLET":

        patch = entries[1]
        numOutlets += 1
        section = "boundaryConditions/outlet" + str(numOutlets)

        addSection(section)        
        newFile[section]["patch"] = patch                
        newFile[section]["type"]  = outletMapping[entries[2]]
        if outletMapping[entries[2]] == 'subsonicPressure':
            newFile[section]["pressure"] = entries[3]
        elif outletMapping[entries[2]] == 'nonReflectingMassFlow':
            newFile[section]["massFlow"] = entries[3]
        elif outletMapping[entries[2]] == 'nonReflectingPressure':
            newFile[section]["pressure"] = entries[3]            
        else:
            logging.error("Unsupported outlet BC type -> %s" % outletMapping[entries[2]])
            exit(1)
    elif varName == "CALC_MEAN_RMS":
        section,name = getNewName(varName)
        addSection(section)
        newFile[section]["startIter"]  = str(entries[1])
        newFile[section]["sampleFreq"] = str(entries[2])
    elif varName == "GRAD_PRESSURE":
        section,name = getNewName(varName)
        addSection(section)
        newFile[section]["enablePressureForcing"] = str(True)
        values = str(entries[1]) + " " + str(entries[2]) + " " + str(entries[3])
        newFile[section]["pressureGrad"] = "'" + values + "'"
    elif varName == "LV_PLANE_NORM":
        section,name = getNewName(varName)
        addSection(section)
        newFile[section]["enableViscousSource"] = str(True)
        values = str(entries[1]) + " " + str(entries[2]) + " " + str(entries[3])
        newFile[section][name] = "'" + values + "'"
    elif varName == "LV_PLANE_P0":
        section,name = getNewName(varName)
        addSection(section)
        newFile[section]["enableViscousSource"] = str(True)
        values = str(entries[1]) + " " + str(entries[2]) + " " + str(entries[3])
        newFile[section][name] = "'" + values + "'"
    elif varName == "LV_PLANE_PINIT":
        section,name = getNewName(varName)
        addSection(section)
        newFile[section]["enableViscousSource"] = str(True)
        values = str(entries[1]) + " " + str(entries[2]) + " " + str(entries[3])
        newFile[section][name] = "'" + values + "'"
    elif varName == "SZ_PLANE_NORM":
        section,name = getNewName(varName)
        addSection(section)
        values = str(entries[1]) + " " + str(entries[2]) + " " + str(entries[3])        
        newFile[section]["normal"] = "'" + values + "'"
    elif varName == "SZ_PLANE_P0":
        section,name = getNewName(varName)
        addSection(section)
        values = str(entries[1]) + " " + str(entries[2]) + " " + str(entries[3])
        newFile[section]["p0"] = "'" + values + "'"
    elif varName == "SZ_PLANE_PINIT":
        section,name = getNewName(varName)
        addSection(section)
        values = str(entries[1]) + " " + str(entries[2]) + " " + str(entries[3])
        newFile[section]["pInit"] = "'" + values + "'"
    elif varName == "SZ_TYPE":
        section,name = getNewName(varName)
        addSection(section)        
        if entries[1] == "0":
            newFile[section]["isEnabled"] = str(True)
            newFile[section]["type"]     = "userDef"
            newFile[section]["density"]  = entries[2]
            uvw = "'" + str(entries[3]) + " " + str(entries[4]) + " " + str(entries[5]) + "'"
            newFile[section]["uvw"] = uvw
            newFile[section]["pressure"] = entries[6]              
        elif entries[1] == "1":
            newFile[section]["isEnabled"] = str(True)
            newFile[section]["type"] = "mixedOut"
            newFile[section]["tolerance"] = entries[2]
        else:
            logging.error("Unknown SZ_TYPE")
    else:
        logging.error("Unknown special input not yet supported -> %s" % varName)
        exit(1)
    

def parseEntry(entry):
    # skip empty lines
    if entry == '': 
        return
    # skip comment lines
    elif entry.startswith("#"):
        return
    else:
        entries = entry.split()

    varName = entries[0]
    numValues = len(entries) - 1

    if isDryRun:
        print(entry)

    if numValues == 0:
        section,newName = getNewName(varName)
        addSection(section)

        if varName == "RESTART_FROM_AUX":
            newFile[section][newName] = "variableP"
        else:
            newFile[section][newName] = str(True)
    elif numValues == 1:
        value = entries[1];
        section,newName = getNewName(varName)
        addSection(section)
        if varName == "RESTART_SERIAL":
            if value == "write":
                newFile[section][newName] = "singleFileWrite"
            elif value == "read":
                newFile[section][newName] = "singleFileRead"
            else:
                logging.error("Unknown value for RESTART_SERIAL conversion")
                exit(1)
        elif varName == "TIME_INTEGRATOR":
            integrators = {}
            integrators["1"] = "forwardEuler"
            integrators["2"] = "rk2"
            integrators["3"] = "rk3"
            integrators["4"] = "rk4"
            integrators["6"] = "rk6"

            if value in integrators:
                newFile[section][newName] = integrators[value]
            else:
                logging.error("Unknown value for TIME_INTEGRATOR conversion => %s " % value)
                exit(1)
        else:
            newFile[section][newName] = str(value)
    else:
        handleMultiValuedInputs(entry)

with open(inFile,"r") as origFile:
    for line in origFile:
        parseEntry(line.strip())

# write new file to stdout
newFile.add_section("boundaryConditions")
newFile["boundaryConditions"]["numWalls"]   = str(numWalls)
newFile["boundaryConditions"]["numInlets"]  = str(numInlets)
newFile["boundaryConditions"]["numOutlets"] = str(numOutlets)

if numScalars > 0:
    newFile.add_section("passiveScalars")
    newFile["passiveScalars"]["numScalars"] = str(numScalars)
    
newFile.write(sys.stdout)

