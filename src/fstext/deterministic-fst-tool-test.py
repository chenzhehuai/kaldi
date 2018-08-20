from ctypes import *

#0 1 4 5 1.4
#0 1 2 5 1.2
#0 1 3 5 1.3
#0 1 1 5 1.1
#1 0 0 0
#0
#fstcompile /tmp/test.fst.txt | fstarcsort -  /tmp/test.fst

fst_lib = CDLL('libkaldi-fstext.so')
fstname='tmp.fst'
max_tok=10
cnt=c_int(0)
fst_lib.allocate.argtypes=[POINTER(c_char)]
STR=(c_char*len(fstname))(*bytes(fstname, "utf-8"))
ret=fst_lib.allocate(STR)
lm_state_next=(c_int*max_tok)()
score=(c_float*max_tok)()
print(ret)
ret=fst_lib.get_next(fst_lib.init(), 2, lm_state_next, score, byref(cnt))
print (ret, ",".join([str(lm_state_next[i]) for i in range(cnt.value)]), ",".join([str(score[i]) for i in range(cnt.value)]), cnt)
ret=fst_lib.get_next(lm_state_next[0], 3, lm_state_next, score, byref(cnt))
print (ret, ",".join([str(lm_state_next[i]) for i in range(cnt.value)]), ",".join([str(score[i]) for i in range(cnt.value)]), cnt)
ret=fst_lib.get_next(lm_state_next[0], 4, lm_state_next, score, byref(cnt))
print (ret, ",".join([str(lm_state_next[i]) for i in range(cnt.value)]), ",".join([str(score[i]) for i in range(cnt.value)]), cnt)
ret=fst_lib.get_next(lm_state_next[0], 9, lm_state_next, score, byref(cnt))
print (ret, ",".join([str(lm_state_next[i]) for i in range(cnt.value)]), ",".join([str(score[i]) for i in range(cnt.value)]), cnt)
fst_lib.free()
