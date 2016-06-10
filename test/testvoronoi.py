import numpy
import scipy.interpolate
from raytrace import voronoi_raytrace
from matplotlib import pyplot as plt
import line

def montecarlo(x, y, z, R, rho, a, b, c, mindistances):
	r = (x**2 + y**2 + z**2)**0.5
	rmax = r.max()
	interpolator = scipy.interpolate.NearestNDInterpolator(numpy.transpose([x, y, z]), rho)
	N = 40000
	t = numpy.logspace(-3, numpy.log10(rmax), N)
	#t = numpy.linspace(0, r.min(), N)
	t = numpy.hstack((numpy.linspace(0, r.min(), 10)[:-1], numpy.logspace(numpy.log10(r.min()), numpy.log10(rmax), N/40)))
	#t = numpy.array(sorted(r))
	points = numpy.transpose([a * t, b * t, c * t])
	densities = interpolator(points)
	#print densities
	#print densities.max(), densities.min()
	for ti, rhoi in zip(t, densities):
		print '  %.3e %.3f' % (rhoi, ti)
	plt.plot(ti, rhoi, '-')
	#print rho.max(), rho.min()
	NH = [numpy.trapz(x=t, y=densities[t >= mindistance]) for mindistance in mindistances]
	print '0...%.2e' % rmax, 'NH:', zip(mindistances, NH)
	return NH

def test():
	data_orig = numpy.load('ray_example.npz')
	data = dict([(k, data_orig[k].astype(numpy.float64)) for k in data_orig.keys()])
	x = data['x']
	y = data['y']
	z = data['z']
	R = data['R']
	rho = data['conversion'][:,0]
	
	vall = data['v'][:1,:]
	
	a, b, c = numpy.copy(vall[:,0]), numpy.copy(vall[:,1]), numpy.copy(vall[:,2])
	mindistances = numpy.copy(data['mindistances'][:1])
	print 'densities:', rho
	print 'MONTE CARLO:'
	result = montecarlo(x, y, z, R, rho, a, b, c, mindistances)
	print result
	
	print 'SEGMENTATION:'
	points = numpy.transpose([x, y, z])
	segments = line.segment(vall[0,:], points)
	for xlo, xhi, i in segments:
		print xlo, xhi, rho[i]
	
	#return result
	
	for i in range(1):
		print 'running...'
		result = voronoi_raytrace(x, y, z, R, rho, a, b, c, mindistances)
	print 'done.'
	print result
	assert result.shape == data['NHtotal'].shape, (result.shape, data['NHtotal'].shape)
	print 'result:', numpy.log10(result)
	print 'reference:', numpy.log10(data['NHtotal'])
	print 'absdiff:', numpy.max(numpy.abs(result - data['NHtotal']))
	print 'reldiff:', numpy.max(numpy.abs((result - data['NHtotal'])/data['NHtotal']))
	assert numpy.allclose(result, data['NHtotal'], rtol=0.0001, atol=1e19)
	

if __name__ == '__main__':
	test()

