NVCC     = nvcc
CXXFLAGS = -std=c++17 -O3
NVCCFLAGS = $(CXXFLAGS) -arch=native

all: compress decompress

compress: compress.cu common.cuh
	$(NVCC) $(NVCCFLAGS) -o compress compress.cu

decompress: decompress.cu common.cuh
	$(NVCC) $(NVCCFLAGS) -o decompress decompress.cu

clean:
	rm -f compress decompress

.PHONY: all clean
