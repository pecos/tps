PROJECT_ROOT = $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

PROGRAM = ex18.p

# LIBRARIES
LIBS = -lmfem

# LIBRARY PATH
LIB_PATHS = /home/marc/Documents/mfem-4.2
# INCLUDES
INCLUDE_PATHS = /home/marc/Documents/mfem-4.2

# SOURCE FILES
SOURCE_FILES = $(wildcard *.cpp)

# HEADER FILES
HEADER_FILES = $(wildcard *.hpp)

# MAIN FILE
MAIN_FILE = ex18.cpp

# OBJECTS
OBJS = $(patsubst %.cpp,%.o,$(SOURCE_FILES))

# COMPILER
CXX = g++

ifeq ($(BUILD_MODE),debug)
	CFLAGS += -Wall -g -std=c++11
else ifeq ($(BUILD_MODE),run)
	CFLAGS += -Wall -O3 -std=c++11
#else
#	$(error Build mode $(BUILD_MODE) not supported by this Makefile)
endif

all: $(PROGRAM)

$(OBJS): $(SOURCE_FILES)
	echo kk
	$(CXX) $(CFLAGS) -c $(patsubst %.o,%.cpp,$@) -I$(INCLUDE_PATHS)

$(PROGRAM): $(OBJS)
	echo kk2
	$(CXX) $(CFLAGS) $(OBJS) -L$(LIB_PATHS) $(LIBS) -o $(PROGRAM)

.PHONY : clean
clean:
	rm -fr $(PROGRAM) *.o
