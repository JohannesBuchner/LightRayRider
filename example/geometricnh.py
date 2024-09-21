import numpy
from numpy import pi, tan, round, log, log10, sin, cos, logical_and, logical_or, arccos, arctan, arctan2
import matplotlib.pyplot as plt
import scipy.interpolate

def to_spherical(cartesian_location):
	(xf, yf, zf) = cartesian_location
	xf = numpy.asarray(xf).reshape((-1,))
	yf = numpy.asarray(yf).reshape((-1,))
	zf = numpy.asarray(zf).reshape((-1,))
	rad = (xf**2+yf**2+zf**2)**0.5
	#phi = numpy.fmod(numpy.arctan2(yf, xf) + 2*pi, pi)
	phi = numpy.arctan2(yf, xf)
	mask = ~(rad == 0)
	theta = numpy.zeros_like(rad)
	theta[mask] = arccos(zf[mask] / rad[mask])
	return (rad, theta, phi)

def to_cartesian(spherical_location):
	(rad, theta, phi) = spherical_location
	xv = rad * sin(theta) * cos(phi)
	yv = rad * sin(theta) * sin(phi)
	zv = rad * cos(theta)
	return (xv, yv, zv)

# assume a sphere 
R = numpy.array([0.5])
r = numpy.array([1.])
x = r*0
y = r*0
rho = x + 1

# ray directions
N = 100000
theta = numpy.arccos(numpy.linspace(0, 1, N))
phi = 0*theta
a, b, c = to_cartesian((1, theta, phi))

# compute intersections
from lightrayrider import sphere_raytrace
NH_torus = sphere_raytrace(r, x, y, R, rho, a, b, c, numpy.array([0.]))[0]
phi = numpy.random.uniform(0, 2*pi, N)
a, b, c = to_cartesian((1, theta, phi))
r[0] = R[0]
NH_sphere = sphere_raytrace(r, x, y, R, rho, a, b, c, numpy.array([0.]))[0]

bins = numpy.logspace(-3, 0, 1000)
plt.hist(NH_torus, bins=bins, density=True, cumulative=True, histtype='step', label='torus')
plt.hist(NH_sphere, bins=bins, density=True, cumulative=True, histtype='step', label='sphere')
#plt.plot(x, 0.5 + 0.5 * x**5)
#plt.plot(x, 0.5 / (1 - x**4))
plt.plot(bins, 1 - 2 * arccos(bins**2)/pi, '--', color='k', lw=3, alpha=0.4)
plt.plot(bins, bins, ':', color='k', lw=3, alpha=0.4)

plt.ylim(1, 0)
plt.xscale('log')
plt.legend(loc='best', prop=dict(size=8))
plt.savefig('geometrynh.pdf', bbox_inches='tight')
plt.close()



