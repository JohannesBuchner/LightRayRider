import numpy
import lightrayrider as raytrace

def to_cartesian(pos):
	(rad, theta, phi) = pos
	sin, cos = numpy.sin, numpy.cos
	xv = rad * sin(theta) * cos(phi)
	yv = rad * sin(theta) * sin(phi)
	zv = rad * cos(theta)
	return (xv, yv, zv)

def test_single():
	xx = numpy.array([0.])
	yy = numpy.array([1.])
	zz = numpy.array([0.])
	RR = numpy.array([0.1])
	rho = numpy.array([5.])
	a = numpy.array([1,0,0.])
	b = numpy.array([0,1,0.])
	c = numpy.array([0,0,1.])
	x = numpy.zeros(3)
	y = x
	z = x
	NHmax = numpy.array([0.5,0.5,0.5])
	t = raytrace.sphere_raytrace_finite(xx, yy, zz, RR, rho, x, y, z, a, b, c, NHmax)
	print(t)
	assert numpy.allclose(t, [-1, 1, -1])
	
	a = numpy.array([0,0,0.])
	b = numpy.array([1,1,1.])
	c = numpy.array([0,0,0.])
	NHmax = numpy.array([1e-10,1 - 1e-10,2])
	t = raytrace.sphere_raytrace_finite(xx, yy, zz, RR, rho, x, y, z, a, b, c, NHmax)
	print(t)
	assert numpy.allclose(t, [0.9, 1.1, -1])

def test_crossing_single():
	xx = numpy.array([0.])
	yy = numpy.array([0.])
	zz = numpy.array([0.])
	RR = numpy.array([1.])
	rho = numpy.array([1.])
	# test at the border
	x = numpy.zeros(3) + 0.99
	y = numpy.zeros(3)
	z = numpy.zeros(3) - 1
	x0, y0, z0 = numpy.copy(x), numpy.copy(y), numpy.copy(z)
	beta = numpy.array([0, 0, 0]) / 180. * numpy.pi
	a, b, c = to_cartesian((1, beta, 0))
	d = numpy.array([0.1, 0.3, 1.1])
	r = raytrace.sphere_raytrace_finite(xx, yy, zz, RR, rho, x, y, z, a, b, c, d)
	print(r)
	assert numpy.isclose(r[0], 0.958932),  r[0]
	assert r[1] == -1, r[1]
	assert r[2] == -1, r[2]

def test_multiple():
	xx = numpy.array([0.,0])
	yy = numpy.array([1.,2])
	zz = numpy.array([0.,0])
	RR = numpy.array([0.1, 0.1])
	rho = numpy.array([5., 5.])
	a = numpy.array([0,0,0.])
	b = numpy.array([1,1,1.])
	c = numpy.array([0,0,0.])
	x = numpy.zeros(3)
	y = x
	z = x
	NHmax = numpy.array([0.999999,1.000001,1.999999])
	t = raytrace.sphere_raytrace_finite(xx, yy, zz, RR, rho, x, y, z, a, b, c, NHmax)
	print(t)
	assert numpy.allclose(t, [1.1, 1.9, 2.1])

def test_multiple_diagonal():
	s3 = numpy.sqrt(3)
	xx = numpy.array([-3,-2,-1,1,2,3.])
	yy = numpy.array([-3,-2,-1,1,2,3.])
	zz = numpy.array([-3,-2,-1,1,2,3.])
	print('xx', xx)
	RR = numpy.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
	rho = numpy.array([5., 5., 5., 5., 5., 5.])
	a = numpy.array([1,1,1.,1]) / s3
	b = numpy.array([1,1,1.,1]) / s3
	c = numpy.array([1,1,1.,1]) / s3
	x = numpy.zeros(4)
	y = x
	z = x
	NHmax = numpy.array([0.99999, 1.99999, 2.9999, 3.01])
	t = raytrace.sphere_raytrace_finite(xx, yy, zz, RR, rho, x, y, z, a, b, c, NHmax)
	print(t)
	assert numpy.allclose(t, [s3+0.1, s3*2+0.1, s3*3+0.1, -1])
	NHmax = numpy.array([0.00001, 1.00001, 2.00001, 3.0001])
	t = raytrace.sphere_raytrace_finite(xx, yy, zz, RR, rho, x, y, z, a, b, c, NHmax)
	print(t)
	assert numpy.allclose(t, [s3-0.1, s3*2-0.1, s3*3-0.1, -1])

def test_angles():
	xx = numpy.array([0.])
	yy = numpy.array([1.])
	zz = numpy.array([0.])
	RR = numpy.array([0.1])
	rho = numpy.array([5.])
	
	phi = numpy.linspace(0, 2*numpy.pi, 400)
	a = numpy.sin(phi)
	b = numpy.cos(phi)
	c = b * 0
	x = numpy.zeros(len(a))
	y = x
	z = x
	NHmax = c * 0 + 0.5
	t = raytrace.sphere_raytrace_finite(xx, yy, zz, RR, rho, x, y, z, a, b, c, NHmax)
	assert t[0] == 1
	assert t[1] > 1
	assert t[10] == -1
	assert t[-10] == -1
	assert t[-2] > 1
	assert t[-1] == 1
	

