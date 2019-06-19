# Author : Levid Rodriguez Santiago

# Compiler to use? (gcc, g++, nvcc, etc)
LINK = nvcc

# Name of binary executable
OUT_FILE = sobelFilter

# flags 
FLAGS = -Wno-deprecated-gpu-targets -O2 -Xcompiler -fopenmp -std=c++11
OPENCV = `pkg-config opencv --cflags --libs`

all: sobelFilter

sobelFilter: sobelFilter.cu
	$(LINK) -o $(OUT_FILE) $(FLAGS) $(OPENCV) $^

clean: 
	rm -f *.o *~ core
