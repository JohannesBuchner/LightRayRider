import numpy
from numpy import pi, sin, cos, arccos, tan
import lightrayrider as raytrace

def cone_raytrace_finite_coords(thetas, rhos, x, y, z, a, b, c, d):
	t = raytrace.cone_raytrace_finite(thetas, rhos, x, y, z, a, b, c, d)
	t[t == -1] = 1e10
	x += a*t
	y += b*t
	z += c*t

def to_spherical(pos):
	(xf, yf, zf) = pos
	xf = numpy.asarray(xf).reshape((-1,))
	yf = numpy.asarray(yf).reshape((-1,))
	zf = numpy.asarray(zf).reshape((-1,))
	rad = (xf**2+yf**2+zf**2)**0.5
	#phi = numpy.fmod(numpy.arctan2(yf, xf) + 2*pi, pi)
	phi = numpy.arctan2(yf, xf)
	theta = numpy.where(rad == 0, 0., arccos(zf / rad))
	return (rad, theta, phi)

def to_cartesian(pos):
	(rad, theta, phi) = pos
	xv = rad * sin(theta) * cos(phi)
	yv = rad * sin(theta) * sin(phi)
	zv = rad * cos(theta)
	return (xv, yv, zv)

def test_central_single_t():
	thetas = numpy.array([45 * pi / 180.])
	rhos = numpy.array([1.0])
	x = numpy.zeros(3)
	y = numpy.zeros(3)
	z = numpy.zeros(3)
	beta = numpy.array([0, 44, 46]) / 180. * numpy.pi
	a, b, c = to_cartesian((1, beta, 0))
	d = numpy.array([0.5, 0.5, 0.5])
	t = raytrace.cone_raytrace_finite(thetas, rhos, x, y, z, a, b, c, d)
	# now x, y, z should be updated
	assert t[0] == -1, r[0]
	assert t[1] == -1, r[1]
	assert t[2] < 1,   r[2]
	assert numpy.isclose(t[2], 0.5),  t[2]

def test_central_single():
	thetas = numpy.array([45 * pi / 180.])
	rhos = numpy.array([1.0])
	x = numpy.zeros(3)
	y = numpy.zeros(3)
	z = numpy.zeros(3)
	beta = numpy.array([0, 44, 46]) / 180. * numpy.pi
	a, b, c = to_cartesian((1, beta, 0))
	d = numpy.array([0.5, 0.5, 0.5])
	cone_raytrace_finite_coords(thetas, rhos, x, y, z, a, b, c, d)
	# now x, y, z should be updated
	r = (x**2 + y**2 + z**2)**0.5
	print(r, x, y, z)
	assert r[0] > 10, r[0]
	assert r[1] > 10, r[1]
	assert r[2] < 1,  r[2]
	assert numpy.isclose(r[2], 0.5),  r[2]
	
	# test bottom part
	x = numpy.zeros(3)
	y = numpy.zeros(3)
	z = numpy.zeros(3)
	beta = numpy.array([90, 90+44, 90+46]) / 180. * numpy.pi
	a, b, c = to_cartesian((1, beta, 0))
	d = numpy.array([0.5, 0.5, 0.5])
	cone_raytrace_finite_coords(thetas, rhos, x, y, z, a, b, c, d)
	# now x, y, z should be updated
	r = (x**2 + y**2 + z**2)**0.5
	print(r, x, y, z)
	assert numpy.isclose(r[0], 0.5),  r[0]
	assert numpy.isclose(r[1], 0.5),  r[1]
	assert r[2] > 10, r[2]

	# test depth
	x = numpy.zeros(3)
	y = numpy.zeros(3)
	z = numpy.zeros(3)
	beta = numpy.array([90, 90, 90]) / 180. * numpy.pi
	a, b, c = to_cartesian((1, beta, 0))
	d = numpy.array([0.1, 0.9, 1.1])
	cone_raytrace_finite_coords(thetas, rhos, x, y, z, a, b, c, d)
	# now x, y, z should be updated
	r = (x**2 + y**2 + z**2)**0.5
	print(r, x, y, z)
	assert numpy.isclose(r[0], 0.1),  r[0]
	assert numpy.isclose(r[1], 0.9),  r[1]
	assert r[2] > 10, r[2]

def test_border_single():
	# test at the border
	thetas = numpy.array([60 * pi / 180.])
	rhos = numpy.array([1.0])
	x = numpy.zeros(3) + 0.99
	y = numpy.zeros(3)
	z = numpy.zeros(3)
	x0, y0, z0 = numpy.copy(x), numpy.copy(y), numpy.copy(z)
	beta = numpy.array([0, 0, 0]) / 180. * numpy.pi
	a, b, c = to_cartesian((1, beta, 0))
	d = numpy.array([0.1, 0.9, 1.1])
	cone_raytrace_finite_coords(thetas, rhos, x, y, z, a, b, c, d)
	# now x, y, z should be updated
	r = ((x-x0)**2 + y**2 + z**2)**0.5
	print(r, x, y, z)
	assert numpy.isclose(r[0], 0.1),  r[0]
	assert r[1] > 10, r[1]
	assert r[2] > 10, r[2]

def test_crossing_single_center():
	# test near the center
	thetas = numpy.array([45 * pi / 180.])
	rhos = numpy.array([1.0])
	x = numpy.zeros(3) + 0.01
	y = numpy.zeros(3)
	z = numpy.zeros(3) - 1
	x0, y0, z0 = numpy.copy(x), numpy.copy(y), numpy.copy(z)
	beta = numpy.array([0, 0, 0]) / 180. * numpy.pi
	a, b, c = to_cartesian((1, beta, 0))
	d = numpy.array([0.01, 0.3, 1.1])
	cone_raytrace_finite_coords(thetas, rhos, x, y, z, a, b, c, d)
	# now x, y, z should be updated
	r = ((x-x0)**2 + (y - y0)**2 + (z-z0)**2)**0.5
	print(r, x, y, z)
	assert numpy.isclose(r[0], 1),  r[0]
	assert r[1] > 10, r[1]
	assert r[2] > 10, r[2]

def test_crossing_single_border():
	# test at the border, where sphere plays the more important role
	thetas = numpy.array([45 * pi / 180.])
	rhos = numpy.array([1.0])
	x = numpy.zeros(3) + 0.99
	y = numpy.zeros(3)
	z = numpy.zeros(3) - 1
	x0, y0, z0 = numpy.copy(x), numpy.copy(y), numpy.copy(z)
	beta = numpy.array([0, 0, 0]) / 180. * numpy.pi
	a, b, c = to_cartesian((1, beta, 0))
	d = numpy.array([0.1, 0.3, 1.1])
	cone_raytrace_finite_coords(thetas, rhos, x, y, z, a, b, c, d)
	# now x, y, z should be updated
	r = ((x-x0)**2 + (y - y0)**2 + (z-z0)**2)**0.5
	print(r, x, y, z)
	assert numpy.isclose(r[0], 0.958932),  r[0]
	assert r[1] > 10, r[1]
	assert r[2] > 10, r[2]
	
def test_noncentral_single():
	# almost central case, result should be the same
	thetas = numpy.array([45 * pi / 180.])
	rhos = numpy.array([1.0])
	x = numpy.zeros(3) + 0.001
	y = numpy.zeros(3)
	z = numpy.zeros(3)
	beta = numpy.array([0, 44, 45]) / 180. * pi
	a, b, c = to_cartesian((1, beta, 0))
	d = numpy.array([0.5, 0.5, 0.5])
	cone_raytrace_finite_coords(thetas, rhos, x, y, z, a, b, c, d)
	# now x, y, z should be updated
	r = ((x - 0.001)**2 + y**2 + z**2)**0.5
	print(r, x, y, z)
	assert r[0] > 10, r[0]
	assert r[1] > 10, r[1]
	assert r[2] < 1,  r[2]
	assert numpy.isclose(r[2], 0.5),  r[2]
	
def test_single_horizontal():
	# cross the sphere and cone
	thetas = numpy.array([45 * pi / 180.])
	rhos = numpy.array([1.0])
	x = numpy.zeros(3) - 2
	y = numpy.zeros(3)
	z = numpy.zeros(3) + 0.001
	a = numpy.ones(3)
	b = numpy.zeros(3)
	c = numpy.zeros(3)
	d = numpy.array([0.001, 1., 1.9])
	cone_raytrace_finite_coords(thetas, rhos, x, y, z, a, b, c, d)
	print(x, y, z)
	xexp = [-1 + 0.001, 0.001*2, 1+0.001*2-0.1]
	assert numpy.allclose(x, xexp, atol=0.001), (x, xexp, x-xexp)
	assert numpy.allclose(y, [0, 0, 0]), y
	assert numpy.allclose(z, [0.001, 0.001, 0.001]), z


def test_central_multiple():
	thetas = numpy.array([45, 60]) * pi / 180.
	rhos = numpy.array([1.0, 2.0])
	x = numpy.zeros(3)
	y = numpy.zeros(3)
	z = numpy.zeros(3)
	beta = numpy.array([44, 46, 70]) / 180. * numpy.pi
	a, b, c = to_cartesian((1, beta, 0))
	d = numpy.array([0.5, 0.5, 0.5])
	cone_raytrace_finite_coords(thetas, rhos, x, y, z, a, b, c, d)
	# now x, y, z should be updated
	r = (x**2 + y**2 + z**2)**0.5
	print(r, x, y, z)
	assert r[0] > 10, r[0]
	assert r[1] < 1, r[1]
	assert numpy.isclose(r[1], 0.5),  r[2]
	assert r[2] < 1, r[2]
	assert numpy.isclose(r[2], 0.25),  r[2]
	
	# test bottom part
	x = numpy.zeros(3)
	y = numpy.zeros(3)
	z = numpy.zeros(3)
	beta = numpy.array([90+20, 90+44, 90+46]) / 180. * numpy.pi
	a, b, c = to_cartesian((1, beta, 0))
	d = numpy.array([0.5, 0.5, 0.5])
	cone_raytrace_finite_coords(thetas, rhos, x, y, z, a, b, c, d)
	# now x, y, z should be updated
	r = (x**2 + y**2 + z**2)**0.5
	print(r, x, y, z)
	assert numpy.isclose(r[0], 0.25),  r[0]
	assert numpy.isclose(r[1], 0.5),  r[1]
	assert r[2] > 10, r[2]

	# test depth
	x = numpy.zeros(3)
	y = numpy.zeros(3)
	z = numpy.zeros(3)
	beta = numpy.array([90, 90, 90]) / 180. * numpy.pi
	a, b, c = to_cartesian((1, beta, 0))
	d = numpy.array([0.1, 0.9, 1.1]) * 2
	cone_raytrace_finite_coords(thetas, rhos, x, y, z, a, b, c, d)
	# now x, y, z should be updated
	r = (x**2 + y**2 + z**2)**0.5
	print(r, x, y, z)
	assert numpy.isclose(r[0], 0.1),  r[0]
	assert numpy.isclose(r[1], 0.9),  r[1]
	assert r[2] > 10, r[2]

def test_speed_central_multiple():
	thetas = numpy.linspace(30, 90, 30) * pi / 180.
	rhos = 10**numpy.linspace(0.1, 2.0, 30)
	N = 1000000
	x = numpy.zeros(N)
	y = numpy.zeros(N)
	z = numpy.zeros(N)
	beta = numpy.random.uniform(0, 90, size=N) / 180. * numpy.pi
	phi = numpy.random.uniform(0, 2*pi, size=N)
	a, b, c = to_cartesian((1, beta, phi))
	d = 10**numpy.random.uniform(-2, 1, size=N) + 0.001
	cone_raytrace_finite_coords(thetas, rhos, x, y, z, a, b, c, d)
	print(x, y, z)
	assert not (x == 0).any()
	assert not (y == 0).any()
	assert not (z == 0).any()

"""
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
	print t
	assert numpy.allclose(t, [-1, 1, -1])
	
	a = numpy.array([0,0,0.])
	b = numpy.array([1,1,1.])
	c = numpy.array([0,0,0.])
	NHmax = numpy.array([1e-10,1 - 1e-10,2])
	t = raytrace.sphere_raytrace_finite(xx, yy, zz, RR, rho, x, y, z, a, b, c, NHmax)
	print t
	assert numpy.allclose(t, [0.9, 1.1, -1])

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
	print t
	assert numpy.allclose(t, [1.1, 1.9, 2.1])

def test_multiple_diagonal():
	s3 = numpy.sqrt(3)
	xx = numpy.array([-3,-2,-1,1,2,3.])
	yy = numpy.array([-3,-2,-1,1,2,3.])
	zz = numpy.array([-3,-2,-1,1,2,3.])
	print 'xx', xx
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
	print t
	assert numpy.allclose(t, [s3+0.1, s3*2+0.1, s3*3+0.1, -1])
	NHmax = numpy.array([0.00001, 1.00001, 2.00001, 3.0001])
	t = raytrace.sphere_raytrace_finite(xx, yy, zz, RR, rho, x, y, z, a, b, c, NHmax)
	print t
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
	
"""
