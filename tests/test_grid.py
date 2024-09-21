import numpy
from math import floor, ceil
from lightrayrider import grid_raytrace_finite

def raytrace_grid_finite(x, v, d):
	(x0, y0, z0), (dx, dy, dz) = x, v
	# call raytrace_grid_finite_c()
	rho = numpy.ones((256, 256, 256))
	x, y, z = numpy.array([x0*1. + 128]), numpy.array([y0*1.+128]), numpy.array([z0*1.+128])
	a, b, c = numpy.array([dx*1.]), numpy.array([dy*1.]), numpy.array([dz*1.])
	NHmax = numpy.array([d*1.])
	r = grid_raytrace_finite(rho, x, y, z, a, b, c, NHmax)
	t = r[0]
	print('distance:', t, (x, y, z), NHmax)
	return x0 + dx * t, y0 + dy * t, z0 + dz * t
	
def test_smallstep():
	x, y, z = numpy.zeros(3)
	dx, dy, dz = 0.99, 0.01, 0
	d = 0.5
	(x1, y1, z1) = raytrace_grid_finite((x, y, z), (dx, dy, dz), d)
	# should land at 5, 0, 0
	assert numpy.allclose(x1, 0.5*0.99), (x1, 0.5*0.99) 
	assert numpy.allclose(y1, 0.5*0.01), (y1, 0.5*0.01) 
	assert numpy.allclose(z1, 0), (z1, 0)

	x, y, z = numpy.zeros(3)
	dx, dy, dz = 0.99, 0.01, 0
	d = 0.5
	(y1, x1, z1) = raytrace_grid_finite((y, x, z), (dy, dx, dz), d)
	# should land at 5, 0, 0
	assert numpy.allclose(x1, 0.5*0.99), (x1, 0.5*0.99) 
	assert numpy.allclose(y1, 0.5*0.01), (y1, 0.5*0.01) 
	assert numpy.allclose(z1, 0), (z1, 0)

	x, y, z = numpy.zeros(3)
	dx, dy, dz = 0.99, 0.01, 0
	d = 0.5
	(z1, y1, x1) = raytrace_grid_finite((z, y, x), (dz, dy, dx), d)
	# should land at 5, 0, 0
	assert numpy.allclose(x1, 0.5*0.99), (x1, 0.5*0.99) 
	assert numpy.allclose(y1, 0.5*0.01), (y1, 0.5*0.01) 
	assert numpy.allclose(z1, 0), (z1, 0)

	x, y, z = numpy.zeros(3)
	dx, dy, dz = 0.99, 0.01, 0
	d = 0.5
	(z1, x1, y1) = raytrace_grid_finite((z, x, y), (dz, dx, dy), d)
	# should land at 5, 0, 0
	assert numpy.allclose(x1, 0.5*0.99), (x1, 0.5*0.99) 
	assert numpy.allclose(y1, 0.5*0.01), (y1, 0.5*0.01) 
	assert numpy.allclose(z1, 0), (z1, 0)

def test_smallstep_offset():
	x, y, z = numpy.ones(3)
	dx, dy, dz = 0.99, 0.01, 0
	d = 0.5
	(x1, y1, z1) = raytrace_grid_finite((x, y, z), (dx, dy, dz), d)
	# should land at 5, 0, 0
	assert numpy.allclose(x1, 1+0.5*0.99), (x1, 1+0.5*0.99) 
	assert numpy.allclose(y1, 1+0.5*0.01), (y1, 1+0.5*0.01) 
	assert numpy.allclose(z1, 1+0), (z1, 1+0)

	x, y, z = numpy.ones(3)
	dx, dy, dz = 0.99, 0.01, 0
	d = 0.5
	(y1, x1, z1) = raytrace_grid_finite((y, x, z), (dy, dx, dz), d)
	# should land at 5, 0, 0
	assert numpy.allclose(x1, 1+0.5*0.99), (x1, 1+0.5*0.99) 
	assert numpy.allclose(y1, 1+0.5*0.01), (y1, 1+0.5*0.01) 
	assert numpy.allclose(z1, 1+0), (z1, 1+0)

	x, y, z = numpy.ones(3)
	dx, dy, dz = 0.99, 0.01, 0
	d = 0.5
	(z1, y1, x1) = raytrace_grid_finite((z, y, x), (dz, dy, dx), d)
	# should land at 5, 0, 0
	assert numpy.allclose(x1, 1+0.5*0.99), (x1, 1+0.5*0.99) 
	assert numpy.allclose(y1, 1+0.5*0.01), (y1, 1+0.5*0.01) 
	assert numpy.allclose(z1, 1+0), (z1, 1+0)


def test_smallstep_negative():
	x, y, z = numpy.zeros(3)
	dx, dy, dz = -0.99, -0.01, 0
	d = 0.5
	(x1, y1, z1) = raytrace_grid_finite((x, y, z), (dx, dy, dz), d)
	# should land at 5, 0, 0
	assert numpy.allclose(x1, -0.5*0.99), (x1, -0.5*0.99) 
	assert numpy.allclose(y1, -0.5*0.01), (y1, -0.5*0.01) 
	assert numpy.allclose(z1, 0), (z1, 0)

	x, y, z = numpy.zeros(3)
	dx, dy, dz = -0.99, -0.01, 0
	d = 0.5
	(y1, x1, z1) = raytrace_grid_finite((y, x, z), (dy, dx, dz), d)
	# should land at 5, 0, 0
	assert numpy.allclose(x1, -0.5*0.99), (x1, -0.5*0.99) 
	assert numpy.allclose(y1, -0.5*0.01), (y1, -0.5*0.01) 
	assert numpy.allclose(z1, 0), (z1, 0)

	x, y, z = numpy.zeros(3)
	dx, dy, dz = -0.99, -0.01, 0
	d = 0.5
	(z1, y1, x1) = raytrace_grid_finite((z, y, x), (dz, dy, dx), d)
	# should land at 5, 0, 0
	assert numpy.allclose(x1, -0.5*0.99), (x1, -0.5*0.99) 
	assert numpy.allclose(y1, -0.5*0.01), (y1, -0.5*0.01) 
	assert numpy.allclose(z1, 0), (z1, 0)


def test_negative():
	x, y, z = numpy.zeros(3)
	dx, dy, dz = -0.99, -0.01, 0
	d = 5
	(x1, y1, z1) = raytrace_grid_finite((x, y, z), (dx, dy, dz), d)
	# should land at 5, 0, 0
	assert numpy.allclose(x1, -5*0.99), (x1, -5*0.99) 
	assert numpy.allclose(y1, -5*0.01), (y1, -5*0.01) 
	assert numpy.allclose(z1, 0), (z1, 0)

	x, y, z = numpy.zeros(3)
	dx, dy, dz = -0.99, -0.01, 0
	d = 5
	(y1, x1, z1) = raytrace_grid_finite((y, x, z), (dy, dx, dz), d)
	# should land at 5, 0, 0
	assert numpy.allclose(x1, -5*0.99), (x1, -5*0.99) 
	assert numpy.allclose(y1, -5*0.01), (y1, -5*0.01) 
	assert numpy.allclose(z1, 0), (z1, 0)

	x, y, z = numpy.zeros(3)
	dx, dy, dz = -0.99, -0.01, 0
	d = 5
	(z1, y1, x1) = raytrace_grid_finite((z, y, x), (dz, dy, dx), d)
	# should land at 5, 0, 0
	assert numpy.allclose(x1, -5*0.99), (x1, -5*0.99)
	assert numpy.allclose(y1, -5*0.01), (y1, -5*0.01)
	assert numpy.allclose(z1, 0), (z1, 0)

def test_reverse():
	# go back
	start = numpy.zeros(3)
	dx, dy, dz = 0.99, 0.01, 0
	d = 0.5
	mid = raytrace_grid_finite(start, (dx, dy, dz), d)
	dx *= -1
	dy *= -1
	dz *= -1
	final = raytrace_grid_finite(mid, (dx, dy, dz), d)
	# should land at 5, 0, 0
	print(start, final)
	assert numpy.allclose(start, final), (start, final) 


	dx, dy, dz = 0.99, 0.01, 0
	d = 1.5
	mid = raytrace_grid_finite(start, (dx, dy, dz), d)
	dx *= -1
	dy *= -1
	dz *= -1
	final = raytrace_grid_finite(mid, (dx, dy, dz), d)
	# should land at 5, 0, 0
	print(start, final)
	assert numpy.allclose(start, final), (start, final) 

	start = numpy.zeros(3)
	dx, dy, dz = 0.5, 0.5, 0.5
	d = 0.5
	mid = raytrace_grid_finite(start, (dx, dy, dz), d)
	dx *= -1
	dy *= -1
	dz *= -1
	final = raytrace_grid_finite(mid, (dx, dy, dz), d)
	# should land at 5, 0, 0
	assert numpy.allclose(start, final), (start, final) 
	

def test_straight():
	x, y, z = numpy.zeros(3)
	dx, dy, dz = 0.99, 0.01, 0
	d = 5.5
	(x1, y1, z1) = raytrace_grid_finite((x, y, z), (dx, dy, dz), d)
	# should land at 5, 0, 0
	assert numpy.allclose(x1, 5.5*0.99), (x1, 5.5*0.99) 
	assert numpy.allclose(y1, 5.5*0.01), (y1, 5.5*0.01) 
	assert numpy.allclose(z1, 0), (z1, 0)

	x, y, z = numpy.zeros(3)
	dx, dy, dz = 0.99, 0.01, 0
	d = 5.5
	(y1, x1, z1) = raytrace_grid_finite((y, x, z), (dy, dx, dz), d)
	# should land at 5, 0, 0
	assert numpy.allclose(x1, 5.5*0.99), (x1, 5.5*0.99) 
	assert numpy.allclose(y1, 5.5*0.01), (y1, 5.5*0.01) 
	assert numpy.allclose(z1, 0), (z1, 0)

	x, y, z = numpy.zeros(3)
	dx, dy, dz = 0.99, 0.01, 0
	d = 5.5
	(z1, y1, x1) = raytrace_grid_finite((z, y, x), (dz, dy, dx), d)
	# should land at 5, 0, 0
	assert numpy.allclose(x1, 5.5*0.99), (x1, 5.5*0.99) 
	assert numpy.allclose(y1, 5.5*0.01), (y1, 5.5*0.01) 
	assert numpy.allclose(z1, 0), (z1, 0)


def test_diagonal():
	x, y, z = numpy.zeros(3)
	dx, dy, dz = 2**0.5, 2**0.5, 0
	d = 5.5
	(x1, y1, z1) = raytrace_grid_finite((x, y, z), (dx, dy, dz), d)
	# should land at 2, 2, 0
	assert numpy.allclose(x1, 5.5*dx), (x1, 5.5*dx)
	assert numpy.allclose(y1, 5.5*dy), (y1, 5.5*dy)
	assert numpy.allclose(z1, 0), (z1, 0)

def test_random():
	for i in range(40):
		x0 = numpy.random.normal(0, 3, size=3)
		dv = numpy.random.normal(size=3)
		dv /= (dv**2).sum()
		d = numpy.random.normal(0, 3)**2
		# expected
		final = x0 + dv * d
		print(x0, dv, d, '--> expect:', final)
		x1 = raytrace_grid_finite(x0, dv, d)
		assert numpy.allclose(final, x1), (final, x1)

def test_random_single():
	x, y, z = numpy.array([125.275611]), numpy.array([127.322529]), numpy.array([128.700102])
	a, b, c = numpy.array([0.22951647]), numpy.array([-0.11787839]), numpy.array([0.19780523])
	print('direction:', a**2 + b**2 + c**2)
	NHmax = numpy.array([63.1791943441])
	rho = numpy.ones((256, 256, 256))
	t = grid_raytrace_finite(rho, x, y, z, a, b, c, NHmax)
	print('distance:', t)
	assert (t > 0).all(), t
	assert numpy.isclose(t, NHmax), (t, NHmax)
	xfinal = x + a * NHmax
	yfinal = y + b * NHmax
	zfinal = z + c * NHmax

def test_single_small1d_pos():
	x, y, z = numpy.array([128.]), numpy.array([128.]), numpy.array([128.])
	a, b, c = numpy.array([0.687837]), numpy.array([0.725799]), numpy.array([0.009826])
	print('direction:', a**2 + b**2 + c**2)
	NHmax = numpy.array([1000.])
	rho = numpy.ones((256, 256, 256))
	t = grid_raytrace_finite(rho, x, y, z, a, b, c, NHmax)
	print('distance:', t)
	assert t == -1, t
	NHmax = numpy.array([10.])
	t = grid_raytrace_finite(rho, x, y, z, a, b, c, NHmax)
	assert (t > 0).all(), t
	assert numpy.isclose(t, NHmax), (t, NHmax)
	
def test_single_small1d_neg():
	x, y, z = numpy.array([128.]), numpy.array([128.]), numpy.array([128.])
	a, b, c = numpy.array([-0.687837]), numpy.array([-0.725799]), numpy.array([0.009826])
	print('direction:', a**2 + b**2 + c**2)
	NHmax = numpy.array([1000.])
	rho = numpy.ones((256, 256, 256))
	t = grid_raytrace_finite(rho, x, y, z, a, b, c, NHmax)
	print('distance:', t)
	assert t == -1, t
	NHmax = numpy.array([10.])
	t = grid_raytrace_finite(rho, x, y, z, a, b, c, NHmax)
	assert (t > 0).all(), t
	assert numpy.isclose(t, NHmax), (t, NHmax)
	
def test_inhomogeneous_single():
	rho = numpy.zeros((256, 256, 256))
	rho[127:130,127:130,127:130] = 1
	rho[128,128,128] = 10
	
	x, y, z = [numpy.array([128.5])]*3
	a, b, c = numpy.array([-0.321689]), numpy.array([-0.901964]), numpy.array([-0.288056])
	print('direction:', a**2 + b**2 + c**2)
	NHmax = numpy.array([(10 + 1 + 1) * 3**0.5])
	t = grid_raytrace_finite(rho, x, y, z, a, b, c, NHmax)
	print('distance:', t)
	assert (t == -1).all(), t


def test_density_coord():
	rho = numpy.zeros((256, 256, 256))
	rho[128,128,128:140] = 1
	x, y, z = [numpy.array([128.]*6)]*3
	a = numpy.array([1., 0., 0., -1., 0., 0.])
	b = numpy.array([0., 1., 0., 0., -1., 0.])
	c = numpy.array([0., 0., 1., 0., 0., -1.])
	NHmax = numpy.array([3.]*6)
	t = grid_raytrace_finite(rho, x, y, z, a, b, c, NHmax)
	t_expect = numpy.array([-1,-1,3., -1,-1,-1])
	assert numpy.allclose(t, t_expect), (t, t_expect)

	rho = numpy.zeros((256, 256, 256))
	rho[128,128:140,128] = 1
	t = grid_raytrace_finite(rho, x, y, z, a, b, c, NHmax)
	t_expect = numpy.array([-1,3.,-1, -1,-1,-1])
	assert numpy.allclose(t, t_expect), (t, t_expect)

	rho = numpy.zeros((256, 256, 256))
	rho[128:140,128,128] = 1
	t = grid_raytrace_finite(rho, x, y, z, a, b, c, NHmax)
	t_expect = numpy.array([3.,-1,-1, -1,-1,-1])
	assert numpy.allclose(t, t_expect), (t, t_expect)

	rho = numpy.zeros((256, 256, 256))
	rho[128,128,120:129] = 1
	t = grid_raytrace_finite(rho, x, y, z, a, b, c, NHmax)
	t_expect = numpy.array([-1,-1,-1, -1,-1,3.])
	assert numpy.allclose(t, t_expect), (t, t_expect)
	
	rho = numpy.zeros((256, 256, 256))
	rho[128,120:129,128] = 1
	t = grid_raytrace_finite(rho, x, y, z, a, b, c, NHmax)
	t_expect = numpy.array([-1,-1,-1, -1,3.,-1])
	assert numpy.allclose(t, t_expect), (t, t_expect)

	rho = numpy.zeros((256, 256, 256))
	rho[120:129,128,128] = 1
	t = grid_raytrace_finite(rho, x, y, z, a, b, c, NHmax)
	t_expect = numpy.array([-1,-1,-1, 3.,-1,-1])
	assert numpy.allclose(t, t_expect), (t, t_expect)



def test_inhomogeneous():
	rho = numpy.zeros((256, 256, 256))
	rho[128,128,128] = 10
	N = 40
	#rho[124:132,124:132,124:132] = 1
	x, y, z = [numpy.zeros(N) + 128.5]*3
	dv = numpy.random.normal(size=(N, 3))
	dv /= (dv**2).sum(axis=1).reshape((-1, 1))**0.5
	a, b, c = dv.transpose()
	a, b, c = numpy.array(a), numpy.array(b), numpy.array(c)
	# call raytrace_grid_finite_c()
	NHmax = 4.99 + numpy.zeros((N))
	t = grid_raytrace_finite(rho, x, y, z, a, b, c, NHmax)
	print('distance:', t)
	assert numpy.allclose(t, 0.499), t
	
	# go one further
	rho[124:132,124:132,124:132] = 1
	rho[128,128,128] = 10
	NHmax = 5.99 + numpy.zeros((N))
	t = grid_raytrace_finite(rho, x, y, z, a, b, c, NHmax)
	print('distance:', t)
	assert (t <= 1.499).all(), t
	assert (t >= 0.599).all(), t
	
	# go even further
	rho = numpy.zeros((256, 256, 256))
	rho[127:130,127:130,127:130] = 1
	rho[128,128,128] = 10
	NHmax = (10 + 1 + 1) * 3**0.5 + numpy.zeros((N))
	#NHmax = 400 + numpy.zeros((20))
	t = grid_raytrace_finite(rho, x, y, z, a, b, c, NHmax)
	print('distance:', t)
	assert (t == -1).all(), t
	

	

