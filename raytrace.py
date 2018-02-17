"""
This file is part of LightRayRider, a fast column density computation tool.

Author: Johannes Buchner (C) 2013-2017
License: AGPLv3

See README and LICENSE file.
"""
from __future__ import print_function, division
import numpy
from ctypes import *
from numpy.ctypeslib import ndpointer
import os

if int(os.environ.get('OMP_NUM_THREADS', '1')) > 1:
	lib = cdll.LoadLibrary(os.path.join(os.path.dirname(__file__), './ray-parallel.so'))
	print("""
  You are using the LightRayRider library, which provides optimized calls for
  photon propagation and column density computations.
  Please cite: Buchner & Bauer (2017), http://adsabs.harvard.edu/abs/2017MNRAS.465.4348B
  Parallelisation enabled.
  """)
else:
	lib = cdll.LoadLibrary(os.path.join(os.path.dirname(__file__), './ray.so'))
	print("""
  You are using the LightRayRider library, which provides optimized calls for
  photon propagation and column density computations.
  Please cite: Buchner & Bauer (2017), http://adsabs.harvard.edu/abs/2017MNRAS.465.4348B
  Parallelisation disabled (use OMP_NUM_THREADS to enable).
  """)



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
	return NHout.reshape((len(mindistances), -1))

lib.sphere_raytrace_count_between.argtypes = [
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
	]

def sphere_raytrace_count_between(xx, yy, zz, RR, rho, a, b, c):
	NHout = numpy.zeros(len(a)) - 1
	lenxx = len(xx)
	lena = len(a)
	r = lib.sphere_raytrace_count_between(xx, yy, zz, RR, lenxx, a, b, c, lena, NHout)
	if r != 0:
		raise Exception("Calculation failed")
	return NHout

lib.grid_raytrace.argtypes = [
	ndpointer(dtype=numpy.float64, ndim=1, flags='C_CONTIGUOUS'), 
	c_int, 
	ndpointer(dtype=numpy.float64, ndim=1, flags='C_CONTIGUOUS'), 
	ndpointer(dtype=numpy.float64, ndim=1, flags='C_CONTIGUOUS'), 
	ndpointer(dtype=numpy.float64, ndim=1, flags='C_CONTIGUOUS'), 
	ndpointer(dtype=numpy.float64, ndim=1, flags='C_CONTIGUOUS'), 
	ndpointer(dtype=numpy.float64, ndim=1, flags='C_CONTIGUOUS'), 
	ndpointer(dtype=numpy.float64, ndim=1, flags='C_CONTIGUOUS'), 
	c_int, 
	ndpointer(dtype=numpy.float64, ndim=1, flags='C_CONTIGUOUS'), 
	]

def grid_raytrace(rho, x, y, z, a, b, c):
	"""
	ray tracing on a grid
	
	Parameters regarding the spheres:
	rho:    double array: density for conversion from length to column density
	 * n:      length of rho
	x:      double array: start vector
	y:      double array: start vector
	z:      double array: start vector
	a:      double array: direction vector
	b:      double array: direction vector
	c:      double array: direction vector
	 * m:      length of a, b, c, x, y, z
	NHout   double array: output; of size m
	"""
	
	lenrho = len(rho)
	#rho_flat = numpy.array(rho.flatten())
	lena = len(a)
	NHout = numpy.zeros(shape=lena) - 1
	r = lib.grid_raytrace(rho, lenrho, x, y, z, a, b, c, lena, NHout)
	if r != 0:
		raise Exception("Calculation failed")
	return NHout


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

lib.sphere_sphere_collisions.argtypes = [
	ndpointer(dtype=numpy.float64, ndim=1, flags='C_CONTIGUOUS'), 
	ndpointer(dtype=numpy.float64, ndim=1, flags='C_CONTIGUOUS'), 
	ndpointer(dtype=numpy.float64, ndim=1, flags='C_CONTIGUOUS'), 
	ndpointer(dtype=numpy.float64, ndim=1, flags='C_CONTIGUOUS'), 
	c_int, 
	c_int, 
	ndpointer(dtype=numpy.float64, ndim=1, flags='C_CONTIGUOUS'), 
	]

def sphere_sphere_collisions(xx, yy, zz, RR, k):
	"""
	marks the first kstop non-intersecting spheres
	
	Parameters regarding the spheres:
	xx:     double array: coordinates
	yy:     double array: coordinates
	zz:     double array: coordinates
	RR:     double array: sphere radius
	k:      int: number of spheres desired
	NHout   double array: output; same size as xx
	"""
	
	lenxx = len(xx)
	NHout = numpy.zeros(lenxx) - 1
	r = lib.sphere_sphere_collisions(xx, yy, zz, RR, lenxx, k, NHout)
	if r != 0:
		raise Exception("Calculation failed")
	return NHout


lib.sphere_raytrace_finite.argtypes = [
	ndpointer(dtype=numpy.float64, ndim=1, flags='C_CONTIGUOUS'), 
	ndpointer(dtype=numpy.float64, ndim=1, flags='C_CONTIGUOUS'), 
	ndpointer(dtype=numpy.float64, ndim=1, flags='C_CONTIGUOUS'), 
	ndpointer(dtype=numpy.float64, ndim=1, flags='C_CONTIGUOUS'), 
	ndpointer(dtype=numpy.float64, ndim=1, flags='C_CONTIGUOUS'), 
	c_int, 
	ndpointer(dtype=numpy.float64, ndim=1, flags='C_CONTIGUOUS'), 
	ndpointer(dtype=numpy.float64, ndim=1, flags='C_CONTIGUOUS'), 
	ndpointer(dtype=numpy.float64, ndim=1, flags='C_CONTIGUOUS'), 
	ndpointer(dtype=numpy.float64, ndim=1, flags='C_CONTIGUOUS'), 
	ndpointer(dtype=numpy.float64, ndim=1, flags='C_CONTIGUOUS'), 
	ndpointer(dtype=numpy.float64, ndim=1, flags='C_CONTIGUOUS'), 
	c_int, 
	ndpointer(dtype=numpy.float64, ndim=1, flags='C_CONTIGUOUS'), 
	ndpointer(dtype=numpy.float64, ndim=1, flags='C_CONTIGUOUS'), 
	]

def sphere_raytrace_finite(xx, yy, zz, RR, rho, x, y, z, a, b, c, NHmax):
	"""
	ray tracing using sphere intersections.
	
	Parameters regarding the spheres:
	xx:     double array: coordinates
	yy:     double array: coordinates
	zz:     double array: coordinates
	RR:     double array: sphere radius
	rho:    double array: density for conversion from length to column density
	 * n:      length of xx, yy, zz, RR, rho
	x:      double array: position vector
	y:      double array: position vector
	z:      double array: position vector
	a:      double array: direction vector
	b:      double array: direction vector
	c:      double array: direction vector
	 * m:      length of a, b, c
	NHmax   double array: stop at this NH
	
	Returns:
	t       double array: end position along direction vector. -1 if infinite
	"""
	
	lenxx = len(xx)
	lena = len(a)
	t = numpy.zeros(shape=lena)
	assert len(b) == lena
	assert len(c) == lena
	assert len(x) == lena
	assert len(y) == lena
	assert len(z) == lena
	assert len(yy) == lenxx
	assert len(zz) == lenxx
	assert len(RR) == lenxx
	assert len(rho) == lenxx
	r = lib.sphere_raytrace_finite(xx, yy, zz, RR, rho, lenxx, x, y, z, a, b, c, lena, NHmax, t)
	if r != 0:
		raise Exception("Calculation failed")
	return t


lib.cone_raytrace_finite.argtypes = [
	ndpointer(dtype=numpy.float64, ndim=1, flags='C_CONTIGUOUS'), 
	ndpointer(dtype=numpy.float64, ndim=1, flags='C_CONTIGUOUS'), 
	c_int, 
	ndpointer(dtype=numpy.float64, ndim=1, flags='C_CONTIGUOUS'), 
	ndpointer(dtype=numpy.float64, ndim=1, flags='C_CONTIGUOUS'), 
	ndpointer(dtype=numpy.float64, ndim=1, flags='C_CONTIGUOUS'), 
	ndpointer(dtype=numpy.float64, ndim=1, flags='C_CONTIGUOUS'), 
	ndpointer(dtype=numpy.float64, ndim=1, flags='C_CONTIGUOUS'), 
	ndpointer(dtype=numpy.float64, ndim=1, flags='C_CONTIGUOUS'), 
	c_int, 
	ndpointer(dtype=numpy.float64, ndim=1, flags='C_CONTIGUOUS'), 
	ndpointer(dtype=numpy.float64, ndim=1, flags='C_CONTIGUOUS'), 
	]

def cone_raytrace_finite(thetas, rhos, x, y, z, a, b, c, d):
	"""
	 * ray tracing through a sphere/cone cuts
	 * 
	 * Sphere radius is 1, each cone angle defines a region of a certain density.
	 * 
	 * This function raytraces from the starting coordinates in the direction
	 * given and compute the column density of the intersecting segments.
	 * Then it will go along the ray in the positive direction until the
	 * column density d is reached. The coordinates of that point are stored into 
	 * (x, y, z)
	 *
	 * Parameters regarding the cones:
	 * thetas: double array: cone opening angle
	 * rhos:   double array: density of each cone from length to column density
	 * n:      number of cones
	 * Parameters regarding the integration direction:
	 * x:      double array: coordinates
	 * y:      double array: coordinates
	 * z:      double array: coordinates
	 * a:      double array: direction vector
	 * b:      double array: direction vector
	 * c:      double array: direction vector
	 * d:      double array: distance to travel
	 * m:      length of (a, b, c) and (x,y,z)
	 * Output:
	 * t       double array: end position along direction vector. -1 if infinite
	"""
	
	n = len(thetas)
	assert len(rhos) == n
	lena = len(a)
	assert len(b) == lena
	assert len(c) == lena
	assert len(x) == lena
	assert len(y) == lena
	assert len(z) == lena
	assert len(d) == lena
	t = numpy.zeros(shape=lena)
	r = lib.cone_raytrace_finite(thetas, rhos, n, x, y, z, a, b, c, lena, d, t)
	if r != 0:
		raise Exception("Calculation failed")
	return t



lib.grid_raytrace_finite.argtypes = [
	ndpointer(dtype=numpy.float64, ndim=1, flags='C_CONTIGUOUS'), 
	c_int, 
	ndpointer(dtype=numpy.float64, ndim=1, flags='C_CONTIGUOUS'), 
	ndpointer(dtype=numpy.float64, ndim=1, flags='C_CONTIGUOUS'), 
	ndpointer(dtype=numpy.float64, ndim=1, flags='C_CONTIGUOUS'), 
	ndpointer(dtype=numpy.float64, ndim=1, flags='C_CONTIGUOUS'), 
	ndpointer(dtype=numpy.float64, ndim=1, flags='C_CONTIGUOUS'), 
	ndpointer(dtype=numpy.float64, ndim=1, flags='C_CONTIGUOUS'), 
	c_int, 
	ndpointer(dtype=numpy.float64, ndim=1, flags='C_CONTIGUOUS'), 
	ndpointer(dtype=numpy.float64, ndim=1, flags='C_CONTIGUOUS'), 
	]

def grid_raytrace_finite(rho, x, y, z, a, b, c, d):
	"""
	ray tracing on a grid
	
	Parameters regarding the spheres:
	rho:    double array: density for conversion from length to column density
	 * n:      length of rho
	x:      double array: start vector
	y:      double array: start vector
	z:      double array: start vector
	a:      double array: direction vector
	b:      double array: direction vector
	c:      double array: direction vector
	 * m:      length of a, b, c, x, y, z
	NHmax   double array: stop at this NH
	
	Returns:
	t       double array: end position along direction vector. -1 if infinite
	"""
	
	lenrho = len(rho)
	#rho_flat = numpy.array(rho.flatten())
	lena = len(a)
	assert len(b) == lena
	assert len(c) == lena
	assert len(x) == lena
	assert len(y) == lena
	assert len(z) == lena
	assert len(d) == lena
	t = numpy.zeros(shape=lena)
	r = lib.grid_raytrace_finite(rho, lenrho, x, y, z, a, b, c, lena, d, t)
	if r != 0:
		raise Exception("Calculation failed")
	return t


