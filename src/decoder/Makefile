all:

EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk
LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS)
LDLIBS += -lnvToolsExt
CUDAOPTS = -lineinfo
#CUDAOPTS += -G -g 


TESTFILES =

OBJFILES = training-graph-compiler.o lattice-simple-decoder.o lattice-faster-decoder.o \
   lattice-faster-online-decoder.o simple-decoder.o faster-decoder.o \
   decoder-wrappers.o     cuda-lattice-decoder_prelink.o cuda-lattice-decoder.o lattice-faster-decoder-cuda.o
#cuda-decoder_prelink.o cuda-decoder.o faster-decoder-cuda.o

LIBNAME = kaldi-decoder

ADDLIBS = ../lat/kaldi-lat.a ../hmm/kaldi-hmm.a \
          ../transform/kaldi-transform.a ../gmm/kaldi-gmm.a \
          ../tree/kaldi-tree.a ../util/kaldi-util.a \
          ../matrix/kaldi-matrix.a ../base/kaldi-base.a

#cuda-decoder_test2.o: cuda-decoder_test2.cu
#	$(CUDATKDIR)/bin/nvcc -rdc=true  -I../ -G -g -O3 -v -c $(CUDA_ARCH) cuda-decoder_test2.cu -lcudadevrt
cuda-decoder_test2: cuda-decoder_test2.cu
	 $(CUDATKDIR)/bin/nvcc -G -g -O3 -arch=sm_70 -rdc=true cuda-decoder_test2.cu -I../ -o cuda-decoder_test2 -lcudadevrt  -lcudart

#cuda-decoder_test2:cuda-decoder_test2.o
#	g++ -g  -O3 -o cuda-decoder_test2 $(CUDA_LDFLAGS)  $(CUDA_LDLIBS)  cuda-decoder_test2.o 


cuda-decoder_test.o: cuda-decoder_test.cu
	$(CUDATKDIR)/bin/nvcc  -G -g -O0 -v -c $(CUDA_ARCH) cuda-decoder_test.cu 

cuda-decoder_test:cuda-decoder_test.o
	g++ -g -O0 -o cuda-decoder_test $(CUDA_LDFLAGS)  $(CUDA_LDLIBS)  -lcuda cuda-decoder_test.o

#cuda-decoder_prelink.o: cuda-decoder.cu cuda-decoder.h
#	$(CUDATKDIR)/bin/nvcc -G -g -O0 -Xptxas -v $(CUDA_ARCH)  -dc -DCUDA_API_PER_THREAD_DEFAULT_STREAM -lineinfo -w -std=c++11 -I.. -I$(OPENFSTINC) -DKALDI_DOUBLEPRECISION=0 -DHAVE_EXECINFO_H=1 -DHAVE_CXXABI_H -DHAVE_ATLAS  -Xcompiler "-msse -msse2 -pthread "  -Xcompiler -fPIC -DHAVE_CUDA -o cuda-decoder_prelink.o cuda-decoder.cu -lcuda
#cuda-decoder.o: cuda-decoder_prelink.o
#	$(CUDATKDIR)/bin/nvcc -G -g -O0 cuda-decoder_prelink.o -v $(CUDA_ARCH)  -dlink -o cuda-decoder.o -Xcompiler "-pthread -fPIC" #-arch=sm_70
thrust_test: thrust_test.cu
	$(CUDATKDIR)/bin/nvcc -g -G -O3 $(CUDA_ARCH) thrust_test.cu -I../ $(CUDA_INCLUDE) -lcudart  -o thrust_test

cub_test3: cub_test3.cu
	$(CUDATKDIR)/bin/nvcc $(CUDAOPTS) $(CUDA_ARCH) cub_test3.cu -I../ $(CUDA_INCLUDE) -lcudart  -o cub_test3
cub_test2: cub_test2.cu
	$(CUDATKDIR)/bin/nvcc -g -G -O3 $(CUDA_ARCH) cub_test2.cu -I../ $(CUDA_INCLUDE) -lcudart  -o cub_test2
cub_test: cub_test.cu
	$(CUDATKDIR)/bin/nvcc -g -O3  -Xptxas   -v -Xcudafe -#  -Xcompiler -ffloat-store -m64  $(CUDA_ARCH) cub_test.cu -I../ $(CUDA_INCLUDE) -lcudart  -o cub_test
cuda-lattice-decoder_prelink.o: cuda-lattice-decoder.cu cuda-lattice-decoder.h
	$(CUDATKDIR)/bin/nvcc  $(CUDAOPTS) -O3  -Xptxas -v $(CUDA_ARCH)  -dc -DCUDA_API_PER_THREAD_DEFAULT_STREAM  -w -std=c++11 -I.. $(CUDA_INCLUDE) -I$(OPENFSTINC) -DKALDI_DOUBLEPRECISION=0 -DHAVE_EXECINFO_H=1 -DHAVE_CXXABI_H -DHAVE_ATLAS  -Xcompiler "-msse -msse2 -pthread "  -Xcompiler -fPIC -DHAVE_CUDA -o cuda-lattice-decoder_prelink.o cuda-lattice-decoder.cu -lcuda  -lcudadevrt
cuda-lattice-decoder.o: cuda-lattice-decoder_prelink.o
	$(CUDATKDIR)/bin/nvcc $(CUDAOPTS) -O3  cuda-lattice-decoder_prelink.o -v $(CUDA_ARCH)  -dlink -o cuda-lattice-decoder.o -Xcompiler "-pthread -fPIC" #-arch=sm_70

include ../makefiles/default_rules.mk
