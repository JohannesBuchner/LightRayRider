import numpy
import os
from lightrayrider import sphere_raytrace

def rootterm(a, b, c, x, y, z, R):
	return (R**2*a**2 + R**2*b**2 + R**2*c**2 - a**2*y**2 - a**2*z**2 + 2*a*b*x*y + 2*a*c*x*z - b**2*x**2 - b**2*z**2 + 2*b*c*y*z - c**2*x**2 - c**2*y**2)

def cross_neg(a, b, c, x, y, z, R, rootterm):
	return (a*x + b*y + c*z - numpy.sqrt(rootterm))/(a**2 + b**2 + c**2)

def cross_pos(a, b, c, x, y, z, R, rootterm):
	return (a*x + b*y + c*z + numpy.sqrt(rootterm))/(a**2 + b**2 + c**2)

def sample_nh(xx, yy, zz, RR, v, mindistances, conversion):
	a = v[:,0].reshape((1, -1))
	b = v[:,1].reshape((1, -1))
	c = v[:,2].reshape((1, -1))
	root = rootterm(a, b, c, xx, yy, zz, RR)
	mask = root >= 0
	print('collisions: %.2f' % (mask.sum(axis=0) * 1.).mean())
	sys.stdout.flush()
	sola = cross_pos(a, b, c, xx, yy, zz, RR, root)
	solb = cross_neg(a, b, c, xx, yy, zz, RR, root)
	#print mask.shape, sola.shape, solb.shape
	lowr = numpy.min([sola, solb], axis=0)
	highr= numpy.max([sola, solb], axis=0)
	assert lowr.shape == sola.shape
	assert highr.shape == solb.shape
	
	seqs = []
	
	for i, mindistance in enumerate(mindistances):
		lowr[lowr < mindistance] = mindistance
		highr[highr < mindistance] = mindistance
		lengths = numpy.where(mask, numpy.abs(highr - lowr), 0)
		NH = lengths * conversion
		NHtotal = NH.sum(axis=0) + 1e18
		seqs.append(NHtotal.tolist())
		#print '** NH=%.1f obscfrac=%.1f%% CTfrac=%.1f%%' % (numpy.log10(NHtotal).mean(), 
		#	100 * (NHtotal>1e22).mean(), 100 * (NHtotal>1e24).mean())
	return seqs

def nh_dist(x, y, z, R, density, mindistances=[0], ):
	"""
	sample the NH from center to random directions
	NH is integrated along the LOS, starting at mindistance [in ckpc/h]
	x, rho, m are the positions, densities and masses of gas clumps,
	which will be assumed to be constant-density spheres
	"""
	# number of H atoms per cm^3
	# assuming everything is in H, which will be a over-estimate of NH
	Msolar_per_kpc3_in_per_cm3 = 4.04e-8
	kpc_in_cm = 3.086e21
	Hdensity = density * Msolar_per_kpc3_in_per_cm3
	conversion = kpc_in_cm * Hdensity.reshape((-1,1))
	
	xx = x.reshape((-1, 1))
	yy = y.reshape((-1, 1))
	zz = z.reshape((-1, 1))
	RR = R.reshape((-1, 1))
	numpy.random.seed(1)

	vall = numpy.random.normal(size=(nsamples, 3))
	assert vall.sum(axis=1).shape == (nsamples,), v.sum(axis=1).shape
	vall /= ((vall**2).sum(axis=1)**0.5).reshape((-1, 1)) # normalize
	assert vall.shape == (nsamples, 3), vall.shape
	
	N = 1
	vs = [vall[i*N:(i+1)*N] for i in range(nsamples / N)]
	allseqs = [sample_nh(xx, yy, zz, RR, v, mindistances, conversion) for v in vs]
	seqs = [[] for mindistance in mindistances]
	
	for chunkseqs in allseqs:
		for i in range(len(mindistances)):
			seqs[i] += chunkseqs[i]
	for i in range(len(mindistances)):
		NHtotal = numpy.array(seqs[i])
		assert NHtotal.shape == (nsamples,), NHtotal.shape
		print('** NH=%.1f obscfrac=%.1f%% CTfrac=%.1f%%' % (numpy.log10(NHtotal).mean(), 
			100 * (NHtotal>1e22).mean(), 100 * (NHtotal>1e24).mean()))
	NHtotal = numpy.array(seqs)
	return NHtotal


def test():
	data_orig = numpy.load(os.path.join(os.path.dirname(__file__), 'ray_example.npz'))
	data = dict([(k, data_orig[k].astype(numpy.float64)) for k in data_orig.keys()])
	x = data['x']
	y = data['y']
	z = data['z']
	R = data['R']
	rho = data['conversion'][:,0]
	
	vall = data['v'] #[:1,:]
	
	a, b, c = numpy.copy(vall[:,0]), numpy.copy(vall[:,1]), numpy.copy(vall[:,2])
	mindistances = numpy.copy(data['mindistances'])
	
	for i in range(1):
		print('running...')
		result = sphere_raytrace(x, y, z, R, rho, a, b, c, mindistances)
	print('done.')
	print(result)
	assert result.shape == data['NHtotal'].shape, (result.shape, data['NHtotal'].shape)
	print('result:', numpy.log10(result))
	print('reference:', numpy.log10(data['NHtotal']))
	print('absdiff:', numpy.max(numpy.abs(result - data['NHtotal'])))
	print('reldiff:', numpy.max(numpy.abs((result - data['NHtotal'])/data['NHtotal'])))
	assert numpy.allclose(result, data['NHtotal'], rtol=0.0001, atol=1e19)
	

if __name__ == '__main__':
	test()

