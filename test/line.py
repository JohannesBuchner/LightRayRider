import matplotlib.pyplot as plt
import numpy
"""
Segment a line into parts, each associated with the nearest point.
"""

def intersection((d1, x1), (d2, x2)):
	return (x2**2 + d2**2 - x1**2 - d1**2) / (2 * (x2 - x1))


def frontier(x, distance, indices, chosen = []):
	#print '  initial:', x[chosen]
	for i in indices:
		# find left & right neighbors
		xdist = x[chosen] - x[i]
		mxdistpos = xdist >= 0
		mxdistneg = xdist < 0

		"""
		# check if on the edges
		if not mxdistpos.any():
			assert False
			chosen.append(i)
			continue
		if not mxdistneg.any():
			assert False
			chosen.append(i)
			continue
		"""
		# find closest
		hii = numpy.argmin(xdist[mxdistpos])
		loi = numpy.argmax(xdist[mxdistneg])
		hi = chosen[numpy.where(mxdistpos)[0][hii]]
		lo = chosen[numpy.where(mxdistneg)[0][loi]]
		# check if shadowed already
		#print '   left: selected', x[lo]
		#print '   right: selected', x[hi]
		xhi = intersection((distance[i], x[i]), (distance[hi], x[hi]))
		xlo = intersection((distance[i], x[i]), (distance[lo], x[lo]))
		#print '   intersection: (%.2f %.2f) (%.2f %.2f) -> %.2f' % (distance[i], x[i], distance[lo], x[lo], xlo)
		#print '   intersection: (%.2f %.2f) (%.2f %.2f) -> %.2f' % (distance[i], x[i], distance[hi], x[hi], xhi)
		# compute the distance along the other's distance
		# find cross-over point between i and hi, and check if distance is larger for lo
		#print '    ', x[i], x[lo], xlo, x[hi], xhi
		#print '   ', distance[i], xhi, x[i], distance[lo], xhi, x[lo]
		#print '   ', distance[i], xlo, x[i], distance[hi], xlo, x[hi]
		if distance[i]**2 + (xhi - x[i])**2 < distance[lo]**2 + (xhi - x[lo])**2 or \
			distance[i]**2 + (xlo - x[i])**2 < distance[hi]**2 + (xlo - x[hi])**2:
			#print '  adding', i, x[i]
			chosen.append(i)
			continue
	return chosen

def linedistance(v, points):
	x = numpy.dot(points, v)
	distancev = x.reshape((-1, 1)) * v.reshape((1, -1)) - points
	distance = (distancev**2).sum(axis=1)**0.5
	return x, distance

def segment(v, points, plot=False):
	x, distance = linedistance(v, points)
	if plot:	
		plt.plot(x, distance, 'x ')

		plt.savefig('line_dist.pdf', bbox_inches='tight')
		plt.savefig('line_dist.png', bbox_inches='tight')
		plt.close()

	# sort by distance
	indices = numpy.argsort(distance).tolist()
	loi = numpy.argmin(x)
	hii = numpy.argmax(x)
	indices.remove(loi)
	indices.remove(hii)
	
	#print x[indices]
	
	chosen = frontier(x, distance, indices, chosen = [loi, hii])
	
	if plot:
		t = numpy.linspace(-20, 20, 400)
		for i in chosen:
			plt.plot(t, (distance[i]**2 + (t - x[i])**2)**0.5, color='grey')

		plt.plot(x, distance, 'x ')
		plt.plot(x[chosen], distance[chosen], 'o')
		plt.xlim(t.min(), t.max())
		plt.ylim(0, distance.max())
		plt.savefig('line_distpareto.pdf', bbox_inches='tight')
		plt.savefig('line_distpareto.png', bbox_inches='tight')
		plt.close()

	# now we can go through and find the points where two are closest
	indices = numpy.argsort(x[chosen])
	chosen = numpy.asarray(chosen)[indices]

	xc = x[chosen]
	dc = distance[chosen]

	xlo = xc[:-1]
	dlo = dc[:-1]
	xhi = xc[1:]
	dhi = dc[1:]

	inf = numpy.inf
	last = 0
	segments = []
	for j in range(len(xlo)):
		# compute cross point
		xmid = intersection((dlo[j], xlo[j]), (dhi[j], xhi[j]))
		#print '     intersection: (%.3f %.3f) (%.3f %.3f) -> %.3f' % (dlo[j], xlo[j], dhi[j], xhi[j], xmid)
		if xmid > last:
			i = chosen[j]
			segments.append((last, xmid, i))
			#print "  seg [%d]: %.3f %.3f" % (i, last, xmid)
			last = xmid
	segments.append((last, inf, chosen[-1]))
	return segments

def nh_segment(segments, densities):
	NH = 0
	for j, (xloi, xhii, i) in enumerate(segments):
		assert xhii >= xloi, (xhii, xloi)
		if xhii < 0:
			continue
		if xloi < 0:
			xloi = 0
		NH += densities[i] * (xhii - xloi)
		#print '   NH contribution:', densities[i], (xhii - xloi), xloi, xhii
	return NH

import scipy.interpolate

def montecarlo1(points, rho, v, mindistances):
	r = (points**2).sum(axis=1)**0.5
	rmax = r.max()
	interpolator = scipy.interpolate.NearestNDInterpolator(points, rho)
	N = 1000
	t = numpy.hstack((numpy.linspace(0, r.min(), 40)[:-1], numpy.logspace(numpy.log10(r.min()), numpy.log10(rmax), N)))
	vpoints = v.reshape((1, -1)) * t.reshape((-1, 1))
	densities = interpolator(vpoints)
	plt.plot(t, densities, '-')
	NHall = [numpy.trapz(x=t, y=densities[t >= mindistance]) for mindistance in mindistances]
	#print '0...%.2e' % rmax, 'NH:', zip(mindistances, NHall)
	return NHall

def montecarlo2(points, rho, v, mindistances):
	r = (points**2).sum(axis=1)**0.5
	rmax = r.max()
	interpolator = scipy.interpolate.NearestNDInterpolator(points, rho)
	N = 10000
	t = numpy.logspace(numpy.log10(r.min()), numpy.log10(rmax), N)
	tlo = numpy.hstack(([0], t[:-1]))
	thi = t
	tmid = (tlo + thi)/2
	tdelta = thi - tlo
	t = numpy.hstack((numpy.linspace(0, r.min(), 40)[:-1], numpy.logspace(numpy.log10(r.min()), numpy.log10(rmax), N/40)))
	vpoints = v.reshape((1, -1)) * tmid.reshape((-1, 1))
	densities = interpolator(vpoints)
	NHall = []
	for mindistance in mindistances:
		mask = tlo >= mindistance
		NH = (tdelta[mask] * densities[mask]).sum()
		NHall.append(NH)
	return NHall



def test(plot=False):
	import time
	numpy.random.seed(3)
	v = numpy.array([1.0, 0, 0])
	t = numpy.linspace(-20, 20, 400)
	linepoints = v * t.reshape((-1, 1))
	points = numpy.random.normal(0, 3, size=(100000, 3))
	densities = 1e23 / (points**2).sum(axis=1)
	
	if plot:
		plt.plot(linepoints[:,0], linepoints[:,1], '-')
		plt.plot(points[:,0], points[:,1], 'x ')

		plt.savefig('line.pdf', bbox_inches='tight')
		plt.savefig('line.png', bbox_inches='tight')
		plt.close()
	
	t0 = time.time()
	segments = segment(v, points, plot=plot)
	
	x, distance = linedistance(v, points)
	chosen = [i for _, _, i in segments]
	if plot:
		plt.plot(x, distance, 'x ')
		plt.plot(x[chosen], distance[chosen], 'o')
	
	xmin = 0
	xmax = ((points**2).sum(axis=1)**0.5).max()
	#print 'choosing segments:'
	segments_limited = []
	for j, (xloi, xhii, i) in enumerate(segments):
		#print '    seg[%d]: %.3f %.3f %.3e' % (j, xloi, xhii, densities[i])
		if not (xhii > xmin):
			#print '        skipping'
			continue
		if not (xloi > xmin):
			xloi = xmin
			if xhii < xloi:
				#print '        skipping'
				continue
		if not (xhii < xmax):
			xhii = xmax
			if xhii < xloi:
				#print '        skipping'
				continue
		segments_limited.append((xloi, xhii, i))
		#xmid = xloi if xloi > 0 else xhii
		xmid = (xloi + xhii) / 2.
		if plot:
			plt.plot([xloi, xhii], [0.1*(j%3+1), 0.1*(j%3+1)], 'x-', linewidth=3)
			plt.plot([xmid, x[i]], [0.04, distance[i]], ':', linewidth=1, color='grey')
	
	NH = nh_segment(segments_limited, densities)
	print 'NH:', NH, time.time() - t0

	if plot:
		plt.xlim(t.min(), t.max())
		plt.ylim(0, distance.max())
		plt.savefig('line_segmented.pdf', bbox_inches='tight')
		plt.savefig('line_segmented.png', bbox_inches='tight')
		plt.close()

	
	print 'MONTE CARLO:'
	t0 = time.time()
	NH = montecarlo2(points, densities, v, [0])
	print 'NH2:', NH, time.time() - t0
	t0 = time.time()
	NH = montecarlo1(points, densities, v, [0])
	print 'NH1:', NH, time.time() - t0
	for j, (xloi, xhii, i) in enumerate(segments_limited):
		plt.plot([xloi, xhii], [densities[i], densities[i]], 'o-', color='k', alpha=0.3)
	plt.savefig('line_montecarlo.pdf', bbox_inches='tight')
	plt.close()
	
	
	x, y, z = numpy.copy(points[:,0]), numpy.copy(points[:,1]), numpy.copy(points[:,2])
	R = y * 0 + 1
	a, b, c = numpy.array([v[0]]), numpy.array([v[1]]), numpy.array([v[2]])
	mindistances = numpy.array([0.0])
	from raytrace import voronoi_raytrace
	t0 = time.time()
	result = voronoi_raytrace(x, y, z, R, densities, a, b, c, mindistances)
	print 'NHC:', result, time.time() - t0
	

if __name__ == '__main__':
	test()

