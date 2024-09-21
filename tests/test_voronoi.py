import os
import numpy
import scipy.interpolate
from lightrayrider import voronoi_raytrace

def montecarlo(x, y, z, R, rho, a, b, c, mindistances):
	r = (x**2 + y**2 + z**2)**0.5
	rmax = r.max()
	interpolator = scipy.interpolate.NearestNDInterpolator(numpy.transpose([x, y, z]), rho)
	N = 400000
	t = numpy.logspace(-3, numpy.log10(rmax), N)
	#t = numpy.linspace(0, r.min(), N)
	#t = numpy.hstack((numpy.linspace(0, r.min(), 10)[:-1], numpy.logspace(numpy.log10(r.min()), numpy.log10(rmax), N/40)))
	#t = numpy.array(sorted(r))
	points = numpy.transpose([a * t, b * t, c * t])
	densities = interpolator(points)
	#print densities
	#print densities.max(), densities.min()
	#for ti, rhoi in zip(t, densities):
	#	print('  %.3e %.3f' % (rhoi, ti))
	#plt.plot(ti, rhoi, '-')
	#print rho.max(), rho.min()
	NH = [numpy.trapz(x=t, y=densities[t >= mindistance]) for mindistance in mindistances]
	print('0...%.2e' % rmax, 'NH:', zip(mindistances, NH))
	return numpy.array(NH)

def test():
	data_orig = numpy.load(os.path.join(os.path.dirname(__file__), 'ray_example.npz'))
	data = dict([(k, data_orig[k].astype(numpy.float64)) for k in data_orig.keys()])
	x = data['x']
	y = data['y']
	z = data['z']
	R = data['R']
	rho = data['conversion'][:,0]
	mindistances = numpy.copy(data['mindistances'][:1])
	print('densities:', rho)
	
	for i in range(3):
		vall = data['v'][i:i+1,:]
		a, b, c = numpy.copy(vall[:,0]), numpy.copy(vall[:,1]), numpy.copy(vall[:,2])
		
		print('MONTE CARLO:', i)
		result = montecarlo(x, y, z, R, rho, a, b, c, mindistances)
		print(result)
	
		#print('SEGMENTATION:')
		#points = numpy.transpose([x, y, z])
		#segments = line.segment(vall[0,:], points)
		#for xlo, xhi, i in segments:
		#	print(xlo, xhi, rho[i])
		#return result
	
		for i in range(1):
			print('running...', i)
			result2 = voronoi_raytrace(x, y, z, R, rho, a, b, c, mindistances).flatten()
		print('done.')
		print(result)
		assert result.shape == result2.shape, (result.shape, result2.shape)
		print('result:', numpy.log10(result))
		print('reference:', numpy.log10(result2))
		print('absdiff:', numpy.max(numpy.abs(result - result2)))
		print('reldiff:', numpy.max(numpy.abs((result - result2)/result2)))
		assert numpy.allclose(result, result2, rtol=0.5, atol=1e20)
	

if __name__ == '__main__':
	test()

