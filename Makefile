NVCC     = nvcc
CXXFLAGS = -std=c++17 -O3
NVCCFLAGS = $(CXXFLAGS) -arch=native

SRCDIR = cucm_gpu
BINDIR = bin

all: $(BINDIR)/compress $(BINDIR)/decompress

$(BINDIR)/compress: $(SRCDIR)/compress.cu $(SRCDIR)/common.cuh | $(BINDIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $(SRCDIR)/compress.cu

$(BINDIR)/decompress: $(SRCDIR)/decompress.cu $(SRCDIR)/common.cuh | $(BINDIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $(SRCDIR)/decompress.cu

$(BINDIR):
	mkdir -p $(BINDIR)

clean:
	rm -f $(BINDIR)/compress $(BINDIR)/decompress

.PHONY: all clean
