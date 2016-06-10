from __future__ import print_function, division
import numpy
import h5py
import os, sys
import json
import matplotlib.pyplot as plt
import scipy.stats
from numpy import log10
from raytrace import voronoi_raytrace

Msolar_per_kpc3_in_per_cm3 = 4.04e-8
kpc_in_cm = 3.086e21

def nh_dist_voronoi(pos, R, density, center, mindistances=[0], nsamples=100):
	"""
	sample the NH from center to random directions
	NH is integrated along the LOS, starting at mindistance [in ckpc/h]
	x, rho, m are the positions, densities and masses of gas clumps,
	which will be assumed to be constant-density spheres
	"""
	x = (pos[:,0] - center[0]).astype(numpy.float64)
	y = (pos[:,1] - center[1]).astype(numpy.float64)
	z = (pos[:,2] - center[2]).astype(numpy.float64)
	
	RR = R.astype(numpy.float64)
	
	# number of H atoms per cm^3
	# assuming everything is in H, which will be a over-estimate of NH
	Hdensity = density * Msolar_per_kpc3_in_per_cm3
	conversion = (kpc_in_cm * Hdensity).astype(numpy.float64)
	
	numpy.random.seed(1)
	vall = numpy.random.normal(size=(nsamples, 3))
	assert vall.sum(axis=1).shape == (nsamples,), v.sum(axis=1).shape
	vall /= ((vall**2).sum(axis=1)**0.5).reshape((-1, 1)) # normalize
	assert vall.shape == (nsamples, 3), vall.shape
	mindistances = numpy.array(mindistances, dtype=numpy.float64)
	a, b, c = numpy.copy(vall[:,0]), numpy.copy(vall[:,1]), numpy.copy(vall[:,2])
	
	NHtotal = voronoi_raytrace(x, y, z, RR, conversion, a, b, c, mindistances)
	assert (conversion >= 0).all(), ((conversion >= 0).mean(), RR[0], a, b, c, mindistances, RR, x, y, z)
	return NHtotal


def wpca(pos, m):
	assert numpy.isfinite(pos).all()
	assert numpy.isfinite(m).all()
	a, b, c = numpy.transpose(pos)
	R = numpy.sqrt(a**2 + b**2 * c**2) + 1
	#mean = (pos * w.reshape((-1,1))) / w.sum()
	pos = pos / R.reshape((-1,1)) # normalize to sphere surface
	assert numpy.isfinite(pos).all()
	indices = numpy.argsort(R)
	weights = m[indices]
	posc = pos[indices]
	Rc = R[indices]
	
	assert numpy.isfinite(posc).all()
	assert numpy.isfinite(weights).all()
	n = (numpy.arange(len(weights)) + 1)
	matrix_sequence = numpy.cumsum([numpy.dot(row.reshape(3, 1), row.reshape(1, 3))*wi for row, wi in zip(posc, weights)], axis=0)

	assert numpy.isfinite(matrix_sequence).all()
	
	alignments = []
	Rout = []
	for i, scatter_matrix in enumerate(matrix_sequence):
		eig_val, eig_vec = numpy.linalg.eig(scatter_matrix)
		Rout.append(Rc[i])
		# relative importance of least important eigenvector
		alignments.append(numpy.abs(numpy.min(eig_val)) / eig_val.sum())
	return Rout, numpy.array(alignments)

def mprofile(pos, m):
	assert numpy.isfinite(pos).all()
	assert numpy.isfinite(m).all()
	a, b, c = numpy.transpose(pos)
	R = numpy.sqrt(a**2 + b**2 * c**2)
	indices = numpy.argsort(R)
	posc = pos[indices]
	mc = m[indices]
	mtot = m.sum()
	Rc = R[indices]
	mfrac = []
	Rout = []
	return Rc, mc.cumsum() / mtot


# from Wilms ISM 
hydrogen_mass_fraction = 0.7103220755035597
helium_mass_fraction = 0.277661268419465
metal_mass_fraction = 0.012016656076975218
ZtoTotal = (helium_mass_fraction / metal_mass_fraction + hydrogen_mass_fraction / metal_mass_fraction + 1)

source = sys.argv[1]
method = sys.argv[2]
for filename in sys.argv[3:]:
	print('reading %s' % filename)
	with h5py.File(filename, 'r') as f:
		g = f['gas']
		if g.keys() == []:
			x = numpy.array([]).reshape((0,3))
			rho = numpy.array([])
			m = numpy.array([])
			m_frac = m
			rho_frac = rho
		else:
			x = g['Coordinates'].value
			rho = g['Density'].value
			assert (rho >= 0).all(), (rho >= 0).mean()
			if 'Mass' in g:
				m = g['Mass'].value
			else:
				m = rho * g['Volume']
	
			if method == 'total':
				frac = 1
			elif method == 'HI':
				frac = g['Mfrac_NeutralHydrogen'].value
			elif method == 'H':
				frac = g['Mfrac_Hydrogen'].value
			elif method == 'Z':
				frac = g['Mfrac_Metals'].value * ZtoTotal * hydrogen_mass_fraction
			elif method == 'ZAGB':
				frac = g['Mfrac_MetalsAGB'].value * ZtoTotal * hydrogen_mass_fraction
			elif method == 'ZSNII':
				frac = g['Mfrac_MetalsSNII'].value * ZtoTotal * hydrogen_mass_fraction
			elif method == 'ZSNIa':
				frac = g['Mfrac_MetalsSNIa'].value * ZtoTotal * hydrogen_mass_fraction
			else:
				raise ValueError('unknown method: "%s". try one of [total, HI, Z]' % method)
			
			if (frac < 0).any():
				# small rounding errors can occur (both in EAGLE and Illustris)
				frac[numpy.logical_and(frac < 0, frac > -1e-6)] = 0
			if (frac < 0).any():
				# larger, should be rare
                                print("WARNING: large negative fraction:", frac.min())
				frac[frac < 0] = 0
			assert (frac >= 0).all(), ((frac >= 0).mean(), frac.min())
			m_frac = m * frac
			rho_frac = rho * frac
		
		sphere_radius = ((m / rho) * 3/4. / numpy.pi)**(1/3.)
		
		if source == 'BH':
			center = f['BH'].attrs['Coordinates']
		elif source == 'densest':
			if len(rho)>0:
				center = x[rho.argmax(),:] + 0.001
			else: # no gas
				center = 0 # does not matter, because no matter
		else:
			continue
		outdata = {}
		
		# NH
		mindistances = [0, 0.1, 1]
		labels=['> %s kpc' % mindistance if mindistance != 0 else 'total' for mindistance in mindistances]
		outdata['mindistances'] = mindistances
		if len(rho_frac) == 0:
			print('        shortcut: NH=1e18')
			NH = numpy.asarray([numpy.zeros(400) + 1e18 for _ in mindistances])
		else:
			NH = nh_dist_voronoi(x, sphere_radius, rho_frac, center, mindistances=mindistances, nsamples=400)
		outdata['NH'] = NH.tolist()
		
		fracs = []
		medians = []
		maxis = []
		for NHi, label in zip(NH, labels):
			obscfrac = (NHi > 1e22).mean()
			obscfrac23 = (NHi > 1e23).mean()
			CTfrac = (NHi > 1e24).mean()
			NHmed = numpy.median(NHi)
			NHmax = float(scipy.stats.mstats.mquantiles(NHi, 0.99))
			print('      ', label, '%.1f%% %.1f NH~%.1f' % (obscfrac * 100, CTfrac * 100, log10(NHmed)))
			fracs.append([obscfrac, CTfrac, obscfrac23])
			medians.append([NHmed, NHmax])
		outdata['NHfracs'] = fracs
		outdata['NHmedians'] = medians
		
		out = '%s_%s_%s.json' % (filename, source, method)
		json.dump(outdata, open(out, 'w'), indent=4)
		
		out = '%s_%s_%s_NH.pdf' % (filename, source, method)
		plt.title('Method: %s' % method)
		hbins = numpy.linspace(20, 25, 40)
		for NHi, label in zip(NH, labels):
			NHi[-(NHi > 1e20)] = 1e20
			plt.hist(numpy.log10(NHi), bins=hbins, label=label, histtype='step')
		plt.xlabel('Column density [log ${cm}^{-2}$]')
		plt.ylabel('number')
		print('plotting histogram into %s' % out)
		plt.legend(loc='best', prop=dict(size=8))
		plt.savefig(out, bbox_inches='tight')
		plt.close()
	

