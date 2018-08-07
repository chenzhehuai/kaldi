from ctypes import *

#0 1 4 5 1.4
#0 1 2 5 1.2
#0 1 3 5 1.3
#0 1 1 5 1.1
#1 0 0 0
#0
#fstcompile /tmp/test.fst.txt | fstarcsort -  /tmp/test.fst

fst_lib = CDLL('libkaldi-fstext.so')
fst_lib.allocate.argtypes=[POINTER(c_char)]
fstname='/tmp/test.fst'
STR=(c_char*len(fstname))(*bytes(fstname, "utf-8"))
ret=fst_lib.allocate(STR)
lm_state_next=c_int(0)
score=c_float(0)
print(ret)
ret=fst_lib.get_next(fst_lib.init(), 2, byref(lm_state_next), byref(score))
print (ret,lm_state_next, score)
ret=fst_lib.get_next(lm_state_next, 4, byref(lm_state_next), byref(score))
print (ret,lm_state_next, score)
ret=fst_lib.get_next(lm_state_next, 3, byref(lm_state_next), byref(score))
print (ret,lm_state_next, score)
ret=fst_lib.get_next(lm_state_next, 9, byref(lm_state_next), byref(score))
print (ret,lm_state_next, score)
fst_lib.free()
