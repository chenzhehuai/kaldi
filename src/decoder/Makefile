all:

EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk
LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS)
LDLIBS += -lnvToolsExt

TESTFILES =

OBJFILES = training-graph-compiler.o lattice-simple-decoder.o lattice-faster-decoder.o \
   lattice-faster-online-decoder.o simple-decoder.o faster-decoder.o \
   decoder-wrappers.o cuda-decoder_prelink.o cuda-decoder.o

LIBNAME = kaldi-decoder

ADDLIBS = ../lat/kaldi-lat.a ../hmm/kaldi-hmm.a \
          ../transform/kaldi-transform.a ../gmm/kaldi-gmm.a \
          ../tree/kaldi-tree.a ../util/kaldi-util.a \
          ../matrix/kaldi-matrix.a ../base/kaldi-base.a


cuda-decoder_prelink.o: cuda-decoder.cu cuda-decoder.h
	$(CUDATKDIR)/bin/nvcc -g -Xptxas -v $(CUDA_ARCH)  -dc -DCUDA_API_PER_THREAD_DEFAULT_STREAM -lineinfo -w -std=c++11 -I.. -I$(OPENFSTINC) -DKALDI_DOUBLEPRECISION=0 -DHAVE_EXECINFO_H=1 -DHAVE_CXXABI_H -DHAVE_ATLAS -I$(ATLASINC) -Xcompiler "-msse -msse2 -pthread -fopenmp" -O0  -Xcompiler -fPIC -DHAVE_CUDA -I${CUDA_HOME}/include -I${CUDA_HOME}/samples/common/inc -I${CUDA_HOME}/include/CL -I${CUDA_HOME}/extras/CUPTI/include -I${CUDA_HOME}/extras/Debugger/include -I${CUDA_HOME}/nvvm/include   -o cuda-decoder_prelink.o cuda-decoder.cu -lcuda
cuda-decoder.o: cuda-decoder_prelink.o
	$(CUDATKDIR)/bin/nvcc cuda-decoder_prelink.o -v $(CUDA_ARCH)  -dlink -o cuda-decoder.o -Xcompiler "-pthread -fPIC" #-arch=sm_70

include ../makefiles/default_rules.mk
