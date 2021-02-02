PROJECT_ROOT = $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

PROGRAM = MulPhyS.p

# LIBRARIES
LIBS =  -lmfem \
	-lHYPRE \
	-lmetis \
	-lrt

# LIBRARY PATH
LIB_PATHS = /home/marc/Public/mfem-4.2 \
	-L/home/marc/Public/hypre/src/hypre/lib \
	-L/home/marc/Public/metis-5.1.0/lib
# INCLUDES -I/home/marc/Public/hypre/include 
INCLUDE_PATHS = /home/marc/Public/mfem-4.2 \
	-I/home/marc/Public/hypre/src/hypre/include \
	-I/home/marc/Public/metis-5.1.0/include

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
	CFLAGS += -Wall -g -O0 -std=c++11 -DDEBUG
else ifeq ($(BUILD_MODE),run)
	CFLAGS += -Wall -O3 -std=c++11
#else
#	$(error Build mode $(BUILD_MODE) not supported by this Makefile)
endif

all: $(PROGRAM)

$(OBJS): $(SOURCE_FILES)
	$(CXX) $(CFLAGS) -c $(patsubst %.o,%.cpp,$@) -I$(INCLUDE_PATHS)

$(PROGRAM): $(OBJS)
	$(CXX) $(CFLAGS) $(OBJS) -L$(LIB_PATHS) $(LIBS) -o $(PROGRAM)

.PHONY : clean
clean:
	rm -fr $(PROGRAM) *.p *.o
