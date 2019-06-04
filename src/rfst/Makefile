
# make "all" the target.
all:

# Disable linking math libs because  not needed here.  Just for compilation speed.
# no, it's now needed for context-fst-test.
# MATHLIB = NONE

EXTRA_CXXFLAGS = -Wno-sign-compare

include ../kaldi.mk

TESTFILES = r-fst-test r-fst-test2

OBJFILES = r-fst.o


RAWLIBNAME = compact_r_fst_compactor-fst

# tree and matrix archives needed for test-context-fst
# matrix archive needed for push-special.
ADDLIBS = ../util/kaldi-util.a ../base/kaldi-base.a  ../fstext/kaldi-fstext.a

include ../makefiles/default_rules.mk
