import matplotlib.pyplot as plt
import numpy

def intersection((d1, x1), (d2, x2)):
	return (x2**2 + d2**2 - x1**2 - d1**2) / (2 * (x2 - x1))
	

"""
Segment a line into parts, each associated with the nearest point.

For this, sample many points and find nearest point for each.
"""

numpy.random.seed(1)

v = numpy.array([1, 0, 0])
t = numpy.linspace(-3, 3, 400)
linepoints = v * t.reshape((-1, 1))
points = numpy.random.normal(0, 3, size=(100, 3))


plt.plot(linepoints[:,0], linepoints[:,1], '-')
plt.plot(points[:,0], points[:,1], 'x ')

plt.savefig('line.pdf', bbox_inches='tight')
plt.savefig('line.png', bbox_inches='tight')
plt.close()

# sort by distance
x = numpy.dot(-points, v)
distancev = -points - x.reshape((-1, 1)) * v.reshape((1, -1))
distance = (distancev**2).sum(axis=1)**0.5

plt.plot(x, distance, 'x ')

plt.savefig('line_dist.pdf', bbox_inches='tight')
plt.savefig('line_dist.png', bbox_inches='tight')
plt.close()

def filter(x, distances, indices, chosen = []):
	for i in indices:
		# find left & right neighbors
		#print i, x[i]
		chosena = numpy.asarray(chosen)
		xdist = x[chosen] - x[i]
		mxdistpos = xdist >= 0
		mxdistneg = xdist < 0

		# check if on the edges
		if not mxdistpos.any():
			chosen.append(i)
			continue
		if not mxdistneg.any():
			chosen.append(i)
			continue

		# find closest
		#print '   x:', x[chosena[mxdistneg]], x[chosena[mxdistpos]]
		hii = numpy.argmin(xdist[mxdistpos])
		loi = numpy.argmax(xdist[mxdistneg])
		hi = chosen[numpy.where(mxdistpos)[0][hii]]
		lo = chosen[numpy.where(mxdistneg)[0][loi]]

		#print '     lo:', loi, lo, x[lo], distance[lo]
		#print '     hi:', hii, hi, x[hi], distance[hi]

		assert x[hi] > x[i], (x[hi], x[i])
		assert x[lo] < x[i], (x[lo], x[i])


		#print '     mi:', i, x[i]
		# check if shadowed already
		xhi = intersection((distance[i], x[i]), (distance[hi], x[hi]))
		xlo = intersection((distance[i], x[i]), (distance[lo], x[lo]))

		# compute the distance along the other's distance
		# find cross-over point between i and hi, and check if distance is larger for lo
		if distance[i]**2 + (xhi - x[i])**2 < distance[lo]**2 + (xhi - x[lo])**2:
			print 'accepting', i, x[i], 'because its distance is closer at xhi than the left one'
			print '    @', xhi, 'from', x[lo], distance[lo]
			print '    X', (distance[i]**2 + (xhi - x[i])**2)**0.5, 'vs', (distance[lo]**2 + (xhi - x[lo])**2)**0.5
			chosen.append(i)
			continue

		if distance[i]**2 + (xlo - x[i])**2 < distance[hi]**2 + (xlo - x[hi])**2:
			print 'accepting', i, x[i], 'because its distance is closer at xlo than the right one'
			chosen.append(i)
			continue
	return chosen
	

indices = numpy.argsort(distance).tolist()
loi = numpy.argmin(x)
hii = numpy.argmax(x)
indices.remove(loi)
indices.remove(hii)

print 'first round...'
chosen = filter(x, distance, indices, chosen = [loi, hii])

## sweep through again
#distance1 = distance[chosen]
#x1 = x[chosen]
#indices = numpy.argsort(x1).tolist()
#loi = indices[0]
#hii = indices[-1]

#print 'second round...'
#chosen = filter(x1, distance1, indices, chosen = chosen[1:-1])
	
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

inf = t.max()
last = -inf
lastpoint = 0
segments = []
# compute cross point
for j in range(len(xlo)):
	xmid = intersection((dlo[j], xlo[j]), (dhi[j], xhi[j]))
	if xmid > last:
		i = chosen[j]
		lastpoint = i
		segments.append((last, xmid, i))
		last = xmid
segments.append((last, inf, chosen[-1]))

#mask = numpy.logical_and(xlo < xmid, xmid < xhi)
#print mask
#for xloi, xhii, xmidi in zip(xlo, xhi, xmid):
#	print xloi, xhii, xmidi
#assert (xhi > xmid).all()
#assert (xlo < xmid < xhi).all()

#segments = [(-inf, xmid[0], chosen[0])]
#for xloi, xhii, i in zip(xmid[:-1], xmid[1:], chosen[1:]):
#	segments.append((xloi, xhii, i))
#segments.append((xmid[-1], inf, chosen[-1]))

plt.plot(x, distance, 'x ')
plt.plot(x[chosen], distance[chosen], 'o')
for j, (xloi, xhii, i) in enumerate(segments):
	print i, xloi, xhii
	plt.plot([xloi, xhii], [0.1*(j%3+1), 0.1*(j%3+1)], 'x-', linewidth=3)
	#xmid = xloi if xloi > 0 else xhii
	xmid = (xloi + xhii) / 2.
	plt.plot([xmid, x[i]], [0.04, distance[i]], ':', linewidth=1, color='grey')

plt.xlim(t.min(), t.max())
plt.ylim(0, dc.max())
plt.savefig('line_segmented.pdf', bbox_inches='tight')
plt.savefig('line_segmented.png', bbox_inches='tight')
plt.close()


