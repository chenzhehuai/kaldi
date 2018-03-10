all:

EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk
LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS)
LDLIBS += -lnvToolsExt
CUDAOPTS =
#CUDAOPTS += -G -g


TESTFILES =

OBJFILES = training-graph-compiler.o lattice-simple-decoder.o lattice-faster-decoder.o \
   lattice-faster-online-decoder.o simple-decoder.o faster-decoder.o \
   decoder-wrappers.o cuda-decoder_prelink.o cuda-decoder.o faster-decoder-cuda.o

LIBNAME = kaldi-decoder

ADDLIBS = ../lat/kaldi-lat.a ../hmm/kaldi-hmm.a \
          ../transform/kaldi-transform.a ../gmm/kaldi-gmm.a \
          ../tree/kaldi-tree.a ../util/kaldi-util.a \
          ../matrix/kaldi-matrix.a ../base/kaldi-base.a

cuda-decoder_test.o: cuda-decoder_test.cu
	$(CUDATKDIR)/bin/nvcc  -G -g -O0 -v -c $(CUDA_ARCH) cuda-decoder_test.cu 

cuda-decoder_test:cuda-decoder_test.o
	g++ -g -O0 -o cuda-decoder_test $(CUDA_LDFLAGS)  $(CUDA_LDLIBS)  -lcuda cuda-decoder_test.o

cuda-decoder_prelink.o: cuda-decoder.cu cuda-decoder.h
	$(CUDATKDIR)/bin/nvcc $(CUDAOPTS) -g -Xptxas -v -arch=sm_70  -dc -DCUDA_API_PER_THREAD_DEFAULT_STREAM -lineinfo -w -std=c++11 -I.. -I/home/jluitjens/git/kaldi/tools/openfst/include -DKALDI_DOUBLEPRECISION=0 -DHAVE_EXECINFO_H=1 -DHAVE_CXXABI_H -DHAVE_ATLAS -I/home/jluitjens/git/kaldi/tools/ATLAS_headers/include -I$(CUDA_INCLUDE) -I.. -I$(OPENFSTINC) -Xcompiler "-msse -msse2 -pthread -fopenmp" -O3  -Xcompiler -fPIC -DHAVE_CUDA    -o cuda-decoder_prelink.o cuda-decoder.cu -lcuda
	
cuda-decoder.o: cuda-decoder_prelink.o
	$(CUDATKDIR)/bin/nvcc $(CUDAOPTS) cuda-decoder_prelink.o -arch=sm_70 -dlink -o cuda-decoder.o -Xcompiler "-pthread -fPIC"


include ../makefiles/default_rules.mk
