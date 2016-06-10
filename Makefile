
CC := gcc
CFLAGS += -fPIC -std=c99 -Wall -lm -Wextra 
CFLAGS += -O3
#CFLAGS += -O0 -g
# compile with the system Python:
#INCDIRS = -I /usr/lib64/python2.7/site-packages/numpy/core/include/  -I/usr/include/python2.7/
# or, instead compile with the Python in Sherpa:
# INCDIRS = -I /opt/ciao-4.6/ots/lib/python2.7/site-packages/numpy/core/include/ -I /opt/ciao-4.6/ots/include/python2.7/
#PYTHON = python


all: ray.so ray-parallel.so
	@echo "done: $@"
allplatforms: ray64.so ray32.so
	@echo "done: $@"

%-parallel.so: %.c
	${CC} ${CFLAGS} -fopenmp -DPARALLEL=1 $< -o $@ -shared

%.so: %.c
	${CC} ${CFLAGS} $< -o $@ -shared
%64.so: %.c 
	${CC} ${CFLAGS} -m64 $< -o $@ -shared
%32.so: %.c 
	${CC} ${CFLAGS} -m32 $< -o $@ -shared

%_BH_Z.json: %
	${PYTHON} irradiate.py BH Z $<
%_BH_ZAGB.json: %
	${PYTHON} irradiate.py BH ZAGB $<
%_BH_ZSNIa.json: %
	${PYTHON} irradiate.py BH ZSNIa $<
%_BH_ZSNII.json: %
	${PYTHON} irradiate.py BH ZSNII $<
%_BH_total.json: %
	${PYTHON} irradiate.py BH total $<
%_BH_HI.json: %
	${PYTHON} irradiate.py BH HI $<
%_BH_H.json: %
	${PYTHON} irradiate.py BH H $<
%_densest_Z.json: %
	${PYTHON} irradiate.py densest Z $<
#%_total.json: %
#	${PYTHON} irradiate.py total $<

clean: 
	rm *.so

.PHONY: all allplatforms clean

