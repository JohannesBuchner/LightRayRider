/***
This file is part of LightRayRider, a fast column density computation tool.

Author: Johannes Buchner (C) 2013-2016
License: AGPLv3

See README and LICENSE file.
***/

#include<stdio.h>
#include<stdlib.h>
#include<assert.h>
//#include<Python.h>
//#include<numpy/arrayobject.h>
#include<math.h>
#ifdef PARALLEL
#include<omp.h>
#endif

#define M_PI           3.14159265358979323846  /* pi */
#define M_PI_2         1.57079632679489661923  /* pi/2 */
#define IFVERBOSE if(0)
#define IFDEBUG if(0)
#define adouble double
#define bdouble double
#define sqr(x) (pow(x,2))

double compute_rootterm(double a, double b, double c, 
	double x, double y, double z, double R) {
	return (sqr(R*a) + sqr(R*b) + sqr(R*c) - sqr(a*y) - sqr(a*z) + 2*a*b*x*y + 2*a*c*x*z - sqr(b*x) - sqr(b*z) + 2*b*c*y*z - sqr(c*x) - sqr(c*y));
}

double cross_neg(double a, double b, double c, 
	double x, double y, double z, double sqrtrootterm) {
	double sqrsum = sqr(a) + sqr(b) + sqr(c);
	return (a*x + b*y + c*z - sqrtrootterm)/sqrsum;
}

double cross_pos(double a, double b, double c, 
	double x, double y, double z, double sqrtrootterm) {
	double sqrsum = sqr(a) + sqr(b) + sqr(c);
	return (a*x + b*y + c*z + sqrtrootterm)/sqrsum;
}


/**
 * ray tracing using sphere intersections
 *
 * Parameters regarding the spheres:
 * xx:     double array: coordinates
 * yy:     double array: coordinates
 * zz:     double array: coordinates
 * RR:     double array: sphere radius
 * rho:    double array: density for conversion from length to column density
 * n:      length of xx, yy, zz, RR
 * Parameters regarding the integration direction:
 * a:      double array: direction vector
 * b:      double array: direction vector
 * c:      double array: direction vector
 * m:      length of a, b, c
 * mindistances double array: only consider intersections beyond these values
 * int l   length of mindistances
 * NHout   double array: output; of size n * l
 * 
 */
int sphere_raytrace(
	const void * xxp, const void * yyp, const void * zzp, const void * RRp, const void * rhop, int n, 
	const void * ap, const void * bp, const void * cp, int m, 
	const void * mindistancesp, int l, const void * NHoutp
) {
	const adouble * mindistances = (adouble*) mindistancesp;
	bdouble * NHout = (bdouble*) NHoutp;
	
	// could use openmp here
	// but actually does not pay off because it is already extremely fast
	#ifdef PARALLEL
	#pragma omp parallel for
	#endif
	for (int i = 0; i < m; i++) { // one ray at a time
		// fprintf(stderr, "ray %d/%d\r", i, m);
		
		const adouble a = ((adouble*) ap)[i];
		const adouble b = ((adouble*) bp)[i];
		const adouble c = ((adouble*) cp)[i];
		
		double NHtotal[l];
		for (int k = 0; k < l; k++) {
			NHtotal[k] = 0.0;
		}
		
		for (int j = 0; j < n; j++) { // one sphere at a time
			const adouble xx = ((adouble*) xxp)[j];
			const adouble yy = ((adouble*) yyp)[j];
			const adouble zz = ((adouble*) zzp)[j];
			const adouble RR = ((adouble*) RRp)[j];
			const adouble rho = ((adouble*) rhop)[j];
			double root = compute_rootterm(a, b, c, xx, yy, zz, RR);
			if (root >= 0) { // actual cut
				double rootsqrt = sqrt(root);
				double lowr = cross_pos(a, b, c, xx, yy, zz, rootsqrt);
				double highr = cross_neg(a, b, c, xx, yy, zz, rootsqrt);
				if (lowr > highr) {
					double tmp = lowr;
					lowr = highr;
					highr = tmp;
				}
				
				for(int k = 0; k < l; k++) {
					double mindistance = mindistances[k];
					if (lowr < mindistance)
						lowr = mindistance;
					if (highr < mindistance)
						highr = mindistance;
					double length = highr - lowr;
					double NHadd = length * rho;
					IFDEBUG fprintf(stderr, "   NH contribution: %e %f %f %f\n", rho, length, lowr, highr);
					NHtotal[k] += NHadd;
				}
			}
		}
		for (int k = 0; k < l; k++) {
			NHout[k * m + i] = NHtotal[k];
		}
	}
	return 0;
}

struct segment {
	double xlo;
	double xhi;
	int i;
};

struct segmentation {
	struct segment * segments;
	int n;
};

struct linedistance {
	double x;
	double distance;
	int i;
};

struct segment single_sphere_intersection(
	const double a, const double b, const double c, 
	const double xx, const double yy, const double zz,
	const double RR 
	) {
	double root = compute_rootterm(a, b, c, xx, yy, zz, RR);
	struct segment seg;
	seg.i = -1;
	if (root >= 0) { // actual cut
		double rootsqrt = sqrt(root);
		seg.xlo = cross_pos(a, b, c, xx, yy, zz, rootsqrt);
		seg.xhi = cross_neg(a, b, c, xx, yy, zz, rootsqrt);
	} else {
		seg.xlo = 0;
		seg.xhi = 0;
	}
	return seg;
}

struct segment single_cone_intersection(
	const double xv, const double yv, const double zv, 
	const double x, const double y, const double z,
	const double theta
	) {
	// compute intersection with cone border
	double tanTheta = pow(tan(M_PI_2 - theta),2);
	double a = pow(zv, 2) - (pow(xv,2) + pow(yv,2)) * tanTheta;
	double b = 2.*z*zv - (2.*x*xv + 2.*y*yv) * tanTheta;
	double c = pow(z, 2) - (pow(x, 2) + pow(y, 2)) * tanTheta;
	double root = pow(b, 2) - 4 * a * c;
	struct segment seg;
	seg.i = -1;
	if (root >= 0) { // actual cut
		seg.xlo = (-b - sqrt(root)) / (2 * a);
		seg.xhi = (-b + sqrt(root)) / (2 * a);
	} else {
		seg.xlo = 0;
		seg.xhi = 0;
	}
	return seg;
}

/* nearest first sorting */
static int cmp_distance(const void *p1, const void *p2) {
	struct linedistance a = *(struct linedistance*) p1;
	struct linedistance b = *(struct linedistance*) p2;
	if (a.distance < b.distance) {
		return -1;
	} else if (a.distance > b.distance) {
		return +1;
	} else {
		return 0;
	}
}

/* line sorting */
static int cmp_x(const void *p1, const void *p2) {
	struct linedistance a = *(struct linedistance*) p1;
	struct linedistance b = *(struct linedistance*) p2;
	if (a.x < b.x) {
		return -1;
	} else if (a.x > b.x) {
		return +1;
	} else {
		return 0;
	}
}

struct linedistance linedistance_create(double vx, double vy, double vz, 
	double px, double py, double pz) {
	/* length along the v vector */
	double x = (px*vx + py*vy + pz*vz);
	/* smallest distance from the v vector */
	double distancex = x * vx - px;
	double distancey = x * vy - py;
	double distancez = x * vz - pz;
	double distance = sqrt(sqr(distancex) + sqr(distancey) + sqr(distancez));
	struct linedistance r = {x, distance, 0};
	return r;
}

/**
 * Compute the point along the line where both points are equidistant
 */
double intersection(const struct linedistance a, const struct linedistance b) {
	double x1 = a.x;
	double d1 = a.distance;
	double x2 = b.x;
	double d2 = b.distance;
	return (sqr(x2) + sqr(d2) - sqr(x1) - sqr(d1)) / (2 * (x2 - x1));
}

/**
 * Select entries in the distances array which form a pareto frontier for the 
 * euclidean distance to the line.
 */
int build_frontier(const struct linedistance * distances, const int n, int * indices, const int imin, const int imax) {
	// i ... index in distances
	// k ... index in indices
	indices[0] = imin;
	indices[1] = imax;
	indices[2] = -1;
	int klen = 2;
	IFDEBUG printf("  leftmost: %.3f\n",  distances[imin].x);
	IFDEBUG printf("  rightmost: %.3f\n", distances[imax].x);
	for(int i = 0; i < n; i++) {
		if (i == imin) continue; // already included
		if (i == imax) continue; // already included
		
		// find left and right neighbors
		int kleft = 0;
		double xdistleft = distances[imin].x - distances[i].x;
		int kright = 1;
		double xdistright = distances[imax].x - distances[i].x;
		for (int k = 2; k < n && indices[k] >= 0; k++) {
			double xdist = distances[indices[k]].x - distances[i].x;
			if (xdist >= 0 && xdist < xdistright) {
				kright = k;
				xdistright = xdist;
			}
			if (xdist < 0 && xdist > xdistleft) {
				kleft = k;
				xdistleft = xdist;
			}
			//printf("    %d: %f %f %d %f %d %f\n", k, distances[indices[k]].x, xdist, kleft, xdistleft, kright, xdistright);
		}
		
		// check if shadowed already by kleft, kright
		int ileft = indices[kleft];
		int iright = indices[kright];
		double xleft = intersection(distances[i], distances[ileft]);
		//IFDEBUG printf("    intersection: (%.2f %.2f) (%.2f %.2f) -> %.2f\n", distances[i].distance, distances[i].x, distances[ileft].distance, distances[ileft].x, xleft);
		double xright = intersection(distances[i], distances[iright]);
		//IFDEBUG printf("    intersection: (%.2f %.2f) (%.2f %.2f) -> %.2f\n", distances[i].distance, distances[i].x, distances[iright].distance, distances[iright].x, xright);
		double di = distances[i].distance;
		double xi = distances[i].x;
		double dileft = distances[ileft].distance;
		double xileft = distances[ileft].x;
		double diright = distances[iright].distance;
		double xiright = distances[iright].x;
		
		//printf("    %f %f %f %f %f\n", xi, xileft, xleft, xiright, xright);
		// compute the distance along the other's distance
		// find cross-over point between i and right, and check if distance is larger for left
		//IFDEBUG printf("    %f %f %f %f\n", dileft, xileft, diright, xiright);
		//IFDEBUG printf("    %f %f %f %f %f %f\n", di, xright, xi, dileft, xright, xileft);
		//IFDEBUG printf("    %f %f %f %f %f %f\n", di, xleft, xi, diright, xleft, xiright);
		if ((sqr(di) + sqr(xright - xi) < sqr(dileft ) + sqr(xright - xileft)) ||
		    (sqr(di) + sqr(xleft  - xi) < sqr(diright) + sqr(xleft - xiright))
		    ) {
		    	IFVERBOSE printf("  adding %d %d %.3f\n", i, distances[i].i, xi);
		    	indices[klen] = i;
		    	klen++;
		    	indices[klen] = -1;
		}
	}
	return klen;
}

/**
 * Segmentation of line
 *
 * a, b, c: direction vector
 * points:
 * xx:     double array: coordinates
 * yy:     double array: coordinates
 * zz:     double array: coordinates
 * n:      length of xx, yy, zz, RR
 */
struct segmentation segments_create(
	const double a, const double b, const double c, 
	const double x, const double y, const double z, 
	const adouble * xx, const adouble * yy, const adouble * zz, const adouble * RR, const int n
	) {
	int * indices = (int*) calloc(n+1, sizeof(int));
	struct linedistance * distances = (struct linedistance *) calloc(n+1, sizeof(struct linedistance));
	double xmin = 1e300;
	int imin = -1;
	double xmax = -1e300;
	int imax = -1;
	
	assert(indices != NULL);
	assert(distances != NULL);
	
	for (int i = 0; i < n; i++) {
		struct linedistance d = linedistance_create(a, b, c, xx[i] - x, yy[i] - y, zz[i] - z);
		d.i = i;
		distances[i] = d;
	}
	// sort by distance
	IFDEBUG fprintf(stderr, "    sorting...\n");
	qsort(distances, n, sizeof(struct linedistance), cmp_distance);

	for (int i = 0; i < n; i++) {
		struct linedistance d = distances[i];
		IFDEBUG fprintf(stderr, "%.2f ", d.x);
		indices[i] = i;
		if (i == 0 || d.x < xmin) {
			imin = i;
			xmin = d.x;
		}
		if (i == 0 || d.x > xmax) {
			imax = i;
			xmax = d.x;
		}
	}
	IFDEBUG fprintf(stderr, "\n");
	IFDEBUG fprintf(stderr, "    building frontier %d, %d...\n", imin, imax);
	int nchosen = build_frontier(distances, n, indices, imin, imax);
	IFDEBUG fprintf(stderr, "    building frontier done, %d chosen\n", nchosen);
	assert(nchosen == 2 || nchosen <= n);
	
	// in indices, 0...nchosen-1 are the chosen points from distances.
	
	// now we can go through and find the points where two are closest
	struct linedistance * distances_chosen = (struct linedistance *) calloc(nchosen, sizeof(struct linedistance));
	for (int i = 0; i < nchosen; i++) {
		// assert(indices[i] >= 0);
		// assert(indices[i] < n);
		distances_chosen[i] = distances[indices[i]];
	}
	IFVERBOSE fprintf(stderr, "freeing distances\n");
	free(distances);
	IFVERBOSE fprintf(stderr, "freeing indices\n");
	free(indices);
	IFVERBOSE fprintf(stderr, "    sorting chosen %d...\n", nchosen);
	qsort(distances_chosen, nchosen, sizeof(struct linedistance), cmp_x);
	IFVERBOSE fprintf(stderr, "    building segments...\n");
	// now create segments
	struct segmentation s;
	s.n = 0;
	s.segments = (struct segment *) calloc(nchosen, sizeof(struct segment));
	assert(s.segments != 0);
	double last = 0; //distances_chosen[0].x - RR[distances_chosen[0].i]; // "-inf"
	IFVERBOSE fprintf(stderr, "last: %f %f %f\n", last, distances_chosen[0].x, RR[distances_chosen[0].i]);
	//last = -20;
	int k = 0;
	// we only integrate until this final point
	struct linedistance distfinal = distances_chosen[nchosen - 1];
	double final = distfinal.x + RR[distfinal.i]; // "+inf"
	for(int j = 1; j < nchosen; j++) {
		// assert(k >= 0);
		// assert(k < nchosen);
		double xmid = intersection(distances_chosen[j-1], distances_chosen[j]);
		IFDEBUG fprintf(stderr, "     intersection: (%.3f %.3f) (%.3f %.3f) -> %.3f\n", 
			distances_chosen[j-1].distance, distances_chosen[j-1].x, 
			distances_chosen[j].distance, distances_chosen[j].x,
			xmid);
		if (xmid > final) {
			struct segment seg = {last, final, distances_chosen[j-1].i};
			IFVERBOSE fprintf(stderr, "  seg [%d]: %.3f %.3f (final)\n", distances_chosen[j-1].i, last, final);
			s.segments[k] = seg;
			last = final;
			k++;
			break;
		}
		if (xmid > last) {
			struct segment seg = {last, xmid, distances_chosen[j-1].i};
			IFVERBOSE fprintf(stderr, "  seg [%d]: %.3f %.3f\n", distances_chosen[j-1].i, last, xmid);
			s.segments[k] = seg;
			last = xmid;
			k++;
		}
	}
	s.n = k;
	IFVERBOSE fprintf(stderr, "  last: %f final: %f\n", last, final);
	// disabled, because can add large distances in unfortunate cases
	if (final > last) {
		// we raytrace not to infinite, but until the most distant 
		// radius. use the last segment's density.
		struct segment seg = {last, final, distfinal.i};
		s.segments[k] = seg;
		s.n = k + 1;
	}
	IFVERBOSE fprintf(stderr, "freeing chosen distances\n");
	free(distances_chosen);
	return s;
}

/**
 * ray tracing using nearest point density
 *
 * Parameters regarding the points:
 * xx:     double array: coordinates
 * yy:     double array: coordinates
 * zz:     double array: coordinates
 * RR:     double array: sphere radius
 * rho:    double array: density for conversion from length to column density
 * n:      length of xx, yy, zz, RR
 * Parameters regarding the integration direction:
 * a:      double array: direction vector
 * b:      double array: direction vector
 * c:      double array: direction vector
 * m:      length of a, b, c
 * mindistances double array: only consider intersections beyond these values
 * int l   length of mindistances
 * NHout   double array: output; of size n * l
 */
int voronoi_raytrace(
	const void * xxp, const void * yyp, const void * zzp, const void * RRp, const void * rhop, int n, 
	const void * ap, const void * bp, const void * cp, int m, 
	const void * mindistancesp, int l, const void * NHoutp
) {
	const adouble * mindistances = (adouble*) mindistancesp;
	bdouble * NHout = (bdouble*) NHoutp;
	
	#ifdef PARALLEL
	#pragma omp parallel for
	#endif
	for (int i = 0; i < m; i++) { // one ray at a time
		IFDEBUG fprintf(stderr, "ray %d/%d\n", i, m);
		fprintf(stderr, "ray %d/%d through %d    \r", i, m, n);
		
		const adouble a = ((adouble*) ap)[i];
		const adouble b = ((adouble*) bp)[i];
		const adouble c = ((adouble*) cp)[i];
		
		
		double NHtotal[l];
		for (int k = 0; k < l; k++) {
			NHtotal[k] = 0.0;
		}
		const adouble * xx = ((adouble*) xxp);
		const adouble * yy = ((adouble*) yyp);
		const adouble * zz = ((adouble*) zzp);
		const adouble * RR = ((adouble*) RRp);
		const adouble * rho = ((adouble*) rhop);
		IFVERBOSE fprintf(stderr, "  segmenting...\n");
		struct segmentation s = segments_create(a, b, c, 0, 0, 0,
			xx, yy, zz, RR, n);
		IFVERBOSE fprintf(stderr, "  segmenting done.\n");
		// now go through the segmentation and sum from mindistances[l] up.
		for (int j = 0; j < s.n; j++) { // one segment at a time
			struct segment seg = s.segments[j];
			double xlo = seg.xlo;
			double xhi = seg.xhi;
			double density = rho[seg.i];
			IFDEBUG fprintf(stderr, "  segment %d/%d: %.3f %.3f %.3e \n", j, s.n, xlo, xhi, density);
			if (xhi < 0)
				continue;
			// assert(seg.i >= 0);
			// assert(seg.i < n);
			for (int k = 0; k < l; k++) {
				double mindistance = mindistances[k];
				if (xhi < mindistance)
					continue;
				if (xlo < mindistance)
					xlo = mindistance;
				double length = xhi - xlo;
				double NHadd = length * density;
				if(!(NHadd >= 0)) {
					fprintf(stderr, "   NH addition odd: %e (%e x %e)\n", NHadd, length, density);
					assert(0);
				}
				//IFDEBUG fprintf(stderr, "   NH contribution: %e %f %f %f\n", density, length, xlo, xhi);
				NHtotal[k] += NHadd;
			}
		}
		free(s.segments);
		for (int k = 0; k < l; k++) {
			IFDEBUG fprintf(stderr, "   NH total: %e\n", NHtotal[k]);
			IFDEBUG if(!(NHtotal[k] >= 0)) {
				fprintf(stderr, "   NH total odd: %e\n", NHtotal[k]);
				assert(0);
			}
			IFDEBUG if(NHtotal[k] >= 1e26) {
				fprintf(stderr, "   NH total high: %e\n", NHtotal[k]);
			}
			NHout[k * m + i] = NHtotal[k];
		}
	}
	return 0;
}



