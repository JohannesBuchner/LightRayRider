"""
This file is part of LightRayRider, a fast column density computation tool.

Author: Johannes Buchner (C) 2013-2016
License: AGPLv3

See README and LICENSE file.
"""
import numpy
from ctypes import *
from numpy.ctypeslib import ndpointer
import os

if int(os.environ.get('OMP_NUM_THREADS', '1')) > 1:
	lib = cdll.LoadLibrary('./ray-parallel.so')
else:
	lib = cdll.LoadLibrary('./ray.so')
lib.sphere_raytrace.argtypes = [
	ndpointer(dtype=numpy.float64, ndim=1, flags='C_CONTIGUOUS'), 
	ndpointer(dtype=numpy.float64, ndim=1, flags='C_CONTIGUOUS'), 
	ndpointer(dtype=numpy.float64, ndim=1, flags='C_CONTIGUOUS'), 
	ndpointer(dtype=numpy.float64, ndim=1, flags='C_CONTIGUOUS'), 
	ndpointer(dtype=numpy.float64, ndim=1, flags='C_CONTIGUOUS'), 
	c_int, 
	ndpointer(dtype=numpy.float64, ndim=1, flags='C_CONTIGUOUS'), 
	ndpointer(dtype=numpy.float64, ndim=1, flags='C_CONTIGUOUS'), 
	ndpointer(dtype=numpy.float64, ndim=1, flags='C_CONTIGUOUS'), 
	c_int, 
	ndpointer(dtype=numpy.float64, ndim=1, flags='C_CONTIGUOUS'), 
	c_int, 
	ndpointer(dtype=numpy.float64, ndim=1, flags='C_CONTIGUOUS'), 
	]

def sphere_raytrace(xx, yy, zz, RR, rho, a, b, c, mindistances):
	"""
	ray tracing using sphere intersections.
	
	Parameters regarding the spheres:
	xx:     double array: coordinates
	yy:     double array: coordinates
	zz:     double array: coordinates
	RR:     double array: sphere radius
	rho:    double array: density for conversion from length to column density
	 * n:      length of xx, yy, zz, RR
	a:      double array: direction vector
	b:      double array: direction vector
	c:      double array: direction vector
	 * m:      length of a, b, c
	mindistances double array: only consider intersections beyond these values
	 * int l   length of mindistances
	NHout   double array: output; of size n * l
	"""
	
	NHout = numpy.zeros(shape=(len(a)*len(mindistances))) - 1
	lenxx = len(xx)
	lena = len(a)
	lenmd = len(mindistances)
	r = lib.sphere_raytrace(xx, yy, zz, RR, rho, lenxx, a, b, c, lena, mindistances, lenmd, NHout)
	if r != 0:
		raise Exception("Calculation failed")
	#for (k = 0; k < l; k++) {
	#	NHout[i * l + k] = NHtotal[k];
	return NHout.reshape((len(mindistances), -1))
	#return NHout

lib.voronoi_raytrace.argtypes = [
	ndpointer(dtype=numpy.float64, ndim=1, flags='C_CONTIGUOUS'), 
	ndpointer(dtype=numpy.float64, ndim=1, flags='C_CONTIGUOUS'), 
	ndpointer(dtype=numpy.float64, ndim=1, flags='C_CONTIGUOUS'), 
	ndpointer(dtype=numpy.float64, ndim=1, flags='C_CONTIGUOUS'), 
	ndpointer(dtype=numpy.float64, ndim=1, flags='C_CONTIGUOUS'), 
	c_int, 
	ndpointer(dtype=numpy.float64, ndim=1, flags='C_CONTIGUOUS'), 
	ndpointer(dtype=numpy.float64, ndim=1, flags='C_CONTIGUOUS'), 
	ndpointer(dtype=numpy.float64, ndim=1, flags='C_CONTIGUOUS'), 
	c_int, 
	ndpointer(dtype=numpy.float64, ndim=1, flags='C_CONTIGUOUS'), 
	c_int, 
	ndpointer(dtype=numpy.float64, ndim=1, flags='C_CONTIGUOUS'), 
	]

def voronoi_raytrace(xx, yy, zz, RR, rho, a, b, c, mindistances):
	"""
	ray tracing using nearest point density.
	
	Parameters regarding the points:
	xx:     double array: coordinates
	yy:     double array: coordinates
	zz:     double array: coordinates
	RR:     double array: sphere radius
	rho:    double array: density for conversion from length to column density
	 * n:      length of xx, yy, zz, RR
	Parameters regarding the integration direction:
	a:      double array: direction vector
	b:      double array: direction vector
	c:      double array: direction vector
	 * m:      length of a, b, c
	mindistances double array: only consider intersections beyond these values
	 * int l   length of mindistances
	NHout   double array: output; of size n * l
	"""
	
	NHout = numpy.zeros(shape=(len(a)*len(mindistances))) - 1
	lenxx = len(xx)
	lena = len(a)
	lenmd = len(mindistances)
	assert xx.shape == (lenxx,), xx.shape
	assert yy.shape == (lenxx,), yy.shape
	assert zz.shape == (lenxx,), zz.shape
	assert RR.shape == (lenxx,), RR.shape
	assert rho.shape == (lenxx,), rho.shape
	assert a.shape == (lena,), a.shape
	assert b.shape == (lena,), b.shape
	assert c.shape == (lena,), c.shape
	assert mindistances.shape == (lenmd,), mindistances.shape
	assert NHout.shape == (lena*lenmd,), NHout.shape
	r = lib.voronoi_raytrace(xx, yy, zz, RR, rho, lenxx, a, b, c, lena, mindistances, lenmd, NHout)
	if r != 0:
		raise Exception("Interpolation failed")
	#for (k = 0; k < l; k++) {
	#	NHout[i * l + k] = NHtotal[k];
	return NHout.reshape((len(mindistances), -1))
	#return NHout


