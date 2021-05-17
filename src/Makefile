PROJECT_ROOT = $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

PROGRAM = MulPhyS.p

# LIBRARIES
LIBS =  -lmfem \
	-lHYPRE \
	-lmetis \
	-lrt

# LIBRARY PATH
LIB_PATHS = -L$(MFEM_LIB) \
	    -L$(HYPRE_LIB) \
	    -L$(METIS_LIB)

# INCLUDES
INCLUDE_PATHS = -I$(MFEM_INC) \
	        -I$(HYPRE_INC) \
	        -I$(METIS_INC)

# SOURCE FILES
SOURCE_FILES = $(wildcard *.cpp)

# HEADER FILES
HEADER_FILES = $(wildcard *.hpp)

# MAIN FILE
MAIN_FILE = main.cpp

# OBJECTS
OBJS = $(patsubst %.cpp,%.o,$(SOURCE_FILES))

# COMPILER
#CXX = g++
CXX = mpicxx

ifeq ($(BUILD_MODE),debug)
	CFLAGS += -Wall -g -O0 -std=c++11 
	DEFINE += -DDEBUG
else ifeq ($(BUILD_MODE),run)
	CFLAGS += -Wall -O3 -std=c++11
#else
#	$(error Build mode $(BUILD_MODE) not supported by this Makefile)
endif

ifeq ($(MASA),true)
	DEFINE += -D_MASA_
	LIBS += -lmasa
	INCLUDE_PATHS += -I$(MASA_INC)
	LIB_PATHS += -L$(MASA_LIB)
endif

all: $(PROGRAM)

$(OBJS): $(SOURCE_FILES)
	$(CXX) $(CFLAGS) $(DEFINE) -c $(patsubst %.o,%.cpp,$@) $(INCLUDE_PATHS)

$(PROGRAM): $(OBJS)
	$(CXX) $(CFLAGS) $(DEFINE) $(OBJS) $(LIB_PATHS) $(LIBS) -o $(PROGRAM)

.PHONY : clean
clean:
	rm -fr $(PROGRAM) *.p *.o
