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
#define sqr(x) (pow(x,2))

/**
 * Helper function for sphere-line intersection.
 * Computes the term under the square root. If it is <= 0, no crossing occured.
 * a, b, c: direction vector of the line, starting from the origin
 * x, y, z: location of the sphere
 * R: radius of the sphere
 */
double compute_rootterm(double a, double b, double c, 
	double x, double y, double z, double R) {
	return (sqr(R*a) + sqr(R*b) + sqr(R*c) - sqr(a*y) - sqr(a*z) + 2*a*b*x*y + 2*a*c*x*z - sqr(b*x) - sqr(b*z) + 2*b*c*y*z - sqr(c*x) - sqr(c*y));
}

/**
 * Helper function for sphere-line intersection.
 * Returns the (lower) point on the line t of the intersection
 * a, b, c: direction vector of the line, starting from the origin
 * x, y, z: location of the sphere
 * sqrtrootterm: square root of what compute_rootterm returned.
 */
double cross_neg(double a, double b, double c, 
	double x, double y, double z, double sqrtrootterm) {
	double sqrsum = sqr(a) + sqr(b) + sqr(c);
	return (a*x + b*y + c*z - sqrtrootterm)/sqrsum;
}

/**
 * Helper function for sphere-line intersection.
 * Returns the (upper) point on the line t of the intersection
 * a, b, c: direction vector of the line, starting from the origin
 * x, y, z: location of the sphere
 * sqrtrootterm: square root of what compute_rootterm returned.
 */
double cross_pos(double a, double b, double c, 
	double x, double y, double z, double sqrtrootterm) {
	double sqrsum = sqr(a) + sqr(b) + sqr(c);
	return (a*x + b*y + c*z + sqrtrootterm)/sqrsum;
}

/**
 * Structure describing a part of a line, between xlo and xhi.
 */
struct segment {
	double xlo;
	double xhi;
	int i;
};


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
static int sphere_raytrace(
	const double * xxp, const double * yyp, const double * zzp, const double * RRp, const double * rhop, int n, 
	const double * ap, const double * bp, const double * cp, int m, 
	const double * mindistancesp, int l, const double * NHoutp
) {
	const double * mindistances = (double*) mindistancesp;
	double * NHout = (double*) NHoutp;
	
	// could use openmp here
	// but actually does not pay off because it is already extremely fast
	#ifdef PARALLEL
	#pragma omp parallel for
	#endif
	for (int i = 0; i < m; i++) { // one ray at a time
		// fprintf(stderr, "ray %d/%d\r", i, m);
		
		const double a = ((double*) ap)[i];
		const double b = ((double*) bp)[i];
		const double c = ((double*) cp)[i];
		
		double NHtotal[l];
		for (int k = 0; k < l; k++) {
			NHtotal[k] = 0.0;
		}
		
		for (int j = 0; j < n; j++) { // one sphere at a time
			const double xx = ((double*) xxp)[j];
			const double yy = ((double*) yyp)[j];
			const double zz = ((double*) zzp)[j];
			const double RR = ((double*) RRp)[j];
			const double rho = ((double*) rhop)[j];
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

/**
 * Count number of spheres lying (partially or completely) between point a,b,c and origin.
 * Stops counting at 1.
 *
 * Parameters regarding the spheres:
 * xx:     double array: coordinates
 * yy:     double array: coordinates
 * zz:     double array: coordinates
 * RR:     double array: sphere radius
 * n:      length of xx, yy, zz, RR
 * Parameters regarding the integration direction:
 * a:      double array: direction vector
 * b:      double array: direction vector
 * c:      double array: direction vector
 * m:      length of a, b, c
 * NHout   double array: output; of size n * l
 * 
 */
static int sphere_raytrace_count_between(
	const double * xxp, const double * yyp, const double * zzp, const double * RRp, int n, 
	const double * ap, const double * bp, const double * cp, int m, 
	const double * NHoutp
) {
	double * NHout = (double*) NHoutp;
	
	// could use openmp here
	// but actually does not pay off because it is already extremely fast
	#ifdef PARALLEL
	#pragma omp parallel for
	#endif
	for (int i = 0; i < m; i++) { // one ray at a time
		#ifndef PARALLEL
		if (i % 100 == 0) fprintf(stderr, "ray %d/%d\r", i, m);
		#endif
		
		const double a = ((double*) ap)[i];
		const double b = ((double*) bp)[i];
		const double c = ((double*) cp)[i];
		
		double NHtotal = 0.0;
		
		for (int j = 0; j < n; j++) { // one sphere at a time
			const double xx = ((double*) xxp)[j];
			const double yy = ((double*) yyp)[j];
			const double zz = ((double*) zzp)[j];
			const double RR = ((double*) RRp)[j];
			//const double rho = ((double*) rhop)[j];
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
				
				if (highr > 1)
					highr = 1;
				if (lowr > 1)
					lowr = 1;
				if (lowr < 0)
					lowr = 0;
				if (highr < 0)
					highr = 0;
				if (highr > lowr) {
					NHtotal = 1;
					break;
				}
			}
		}
		NHout[i] = NHtotal;
	}
	return 0;
}

struct segmentation {
	struct segment * segments;
	int n;
};

struct linedistance {
	double x;
	double distance;
	int i;
};

/**
 * Returns the part of a line that intersects with a given sphere.
 * a, b, c: direction vector of the line, starting from the origin
 * xx, yy, zz: location of the sphere
 * RR: radius of the sphere
 * A empty segment (xlo=xhi=0) is returned if no crossing occured. 
 */
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
		seg.xlo = cross_neg(a, b, c, xx, yy, zz, rootsqrt);
		seg.xhi = cross_pos(a, b, c, xx, yy, zz, rootsqrt);
	} else {
		seg.xlo = 0;
		seg.xhi = 0;
	}
	return seg;
}

/**
 * Returns the part of a line that intersects with a given cone.
 * xv, yv, zv: direction vector of the line, starting from the origin
 * x, y, z: origin of the cone
 * theta: half-opening angle of the cone
 * A empty segment (xlo=xhi=0) is returned if no crossing occured. 
 */
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

/* nearest first sorting */
static int cmp_segments(const void *p1, const void *p2) {
	struct segment a = *(struct segment*) p1;
	struct segment b = *(struct segment*) p2;
	if (a.xlo < b.xlo) {
		return -1;
	} else if (a.xlo > b.xlo) {
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

/* double value sorting */
static int cmp_double(const void * p1, const void *p2) {
	const double a = *(double*) p1;
	const double b = *(double*) p2;
	if (a < b) {
		return -1;
	} else if (a > b) {
		return +1;
	} else {
		return 0;
	}
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
 * n:      length of xx, yy, zz, RR, rho
 * Parameters regarding the integration direction:
 * x:      double array: position vector
 * y:      double array: position vector
 * z:      double array: position vector
 * a:      double array: direction vector
 * b:      double array: direction vector
 * c:      double array: direction vector
 * m:      length of a, b, c
 * NHmax   double array: stop at this NH
 * Output:
 * t       double array: end position along direction vector
 */
static int sphere_raytrace_finite(
	const double * xxp, const double * yyp, const double * zzp, const double * RRp, const double * rhop, int n, 
	const double * xp, const double * yp, const double * zp, const double * ap, const double * bp, const double * cp, int m, 
	const double * NHmaxp, const double * tp
) {
	double * NHmax = (double*) NHmaxp;
	double * rho = (double*) rhop;
	double * t = (double*) tp;
	
	#ifdef PARALLEL
	#pragma omp parallel for
	#endif
	for (int i = 0; i < m; i++) { // one ray at a time
		const double a = ((double*) ap)[i];
		const double b = ((double*) bp)[i];
		const double c = ((double*) cp)[i];
		const double x = ((double*) xp)[i];
		const double y = ((double*) yp)[i];
		const double z = ((double*) zp)[i];
		
		// collect spheres into segmentation
		
		// create segments
		struct segmentation s;
		s.n = 0;
		s.segments = (struct segment *) calloc(n, sizeof(struct segment));
		assert(s.segments != 0);
		
		// for each sphere we should compute tmin, tmax of cut
		for (int j = 0; j < n; j++) { // one sphere at a time
			const double xx = ((double*) xxp)[j] - x;
			const double yy = ((double*) yyp)[j] - y;
			const double zz = ((double*) zzp)[j] - z;
			const double RR = ((double*) RRp)[j];
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
				if (lowr < 0) {
					lowr = 0;
				}
				if (highr <= lowr) {
					continue;
				}
				
				IFDEBUG printf("RAY %d cuts sphere %d \n", i, j);
				struct segment seg = {lowr, highr, j};
				s.segments[s.n] = seg;
				s.n++;
			}
		}
		// sort segments
		qsort(s.segments, s.n, sizeof(struct segment), cmp_segments);
		
		double NHleft = NHmax[i];
		// decrease NH until done
		for(int k = 0; k < s.n; k++) {
			IFDEBUG printf("  - sphere %d \n", s.segments[k].i);
			double length = s.segments[k].xhi - s.segments[k].xlo;
			double NHk = length * rho[s.segments[k].i];
			if (NHleft > NHk) {
				// we pass through this sphere and decrease NH
				IFDEBUG printf("     passing through %f -> %f \n", NHleft, NHleft - NHk);
				NHleft -= NHk;
			} else {
				// we stop inside this sphere
				// compute where we stop
				t[i] = NHleft / rho[s.segments[k].i] + s.segments[k].xlo;
				IFDEBUG printf("     stopping inside %f -> t=%f \n", NHleft, t[i]);
				NHleft = -1;
				break;
			}
		}
		free(s.segments);
		if (NHleft > 0) {
			// we go to infinity ...
			IFDEBUG printf("     went to inf\n");
			t[i] = -1;
		}
	}
	return 0;
}

#define min(a, b) ((a) > (b) ? (b) : (a))

/**
 * ray tracing on a grid
 *
 * Parameters regarding the uniform, cartesian grid:
 * rho:    double array: density for conversion from length to column density
 * n:      length of rho
 * Parameters regarding the integration direction:
 * x:      double array: position vector
 * y:      double array: position vector
 * z:      double array: position vector
 * a:      double array: direction vector
 * b:      double array: direction vector
 * c:      double array: direction vector
 * m:      length of a, b, c
 * NHmax   double array: stop at this NH
 * Output:
 * t       double array: end position along direction vector
 */
static int grid_raytrace_finite(
	const double * rhop, int n, 
	const double * xp, const double * yp, const double * zp, const double * ap, const double * bp, const double * cp, int m, 
	const double * NHmaxp, const double * tp
) {
	double * NHmax = (double*) NHmaxp;
	double * rhoarr = (double*) rhop;
	double * tout = (double*) tp;
	
	#ifdef PARALLEL
	#pragma omp parallel for
	#endif
	for (int i = 0; i < m; i++) { // one ray at a time
		const double dx = ((double*) ap)[i];
		const double dy = ((double*) bp)[i];
		const double dz = ((double*) cp)[i];
		const double x0 = ((double*) xp)[i];
		const double y0 = ((double*) yp)[i];
		const double z0 = ((double*) zp)[i];
		
		double d = NHmax[i];
		
		const double dt_dx = 1. / dx;
		const double dt_dy = 1. / dy;
		const double dt_dz = 1. / dz;
		
		int x = x0;
		int y = y0;
		int z = z0;
		int x_inc, y_inc, z_inc;
		double t_next_x, t_next_y, t_next_z;
		int sign_x = 1, sign_y = 1, sign_z = 1;
		
		if (dx == 0) {
			x_inc = 0;
			t_next_x = dt_dx; // inf
		} else if (dx > 0) {
			x_inc = 1;
			t_next_x = (floor(x0) + 1 - x0) / dx;
		} else {
			x_inc = -1;
			t_next_x = (x0 - floor(x0)) / dx;
			sign_x = -1;
		}
		if (dy == 0) {
			y_inc = 0;
			t_next_y = dt_dy; // inf
		} else if (dy > 0) {
			y_inc = 1;
			t_next_y = (floor(y0) + 1 - y0) / dy;
		} else {
			y_inc = -1;
			t_next_y = (y0 - floor(y0)) / dy;
			sign_y = -1;
		}
		if (dz == 0) {
			z_inc = 0;
			t_next_z = dt_dz; // inf
		} else if (dz > 0) {
			z_inc = 1;
			t_next_z = (floor(z0) + 1 - z0) / dz;
		} else {
			z_inc = -1;
			t_next_z = (z0 - floor(z0)) / dz;
			sign_z = -1;
		}
		IFDEBUG printf("ray       : %f %f %f | %f %f %f | %f\n", x0, y0, z0, dx, dy, dz, d);
		IFDEBUG printf("step sizes: %f %f %f\n", t_next_x, t_next_y, t_next_z);
		double t;
		double last_t = 0;
		int last_x = x;
		int last_y = y;
		int last_z = z;
		if (x < 0 || x >= n || y < 0 || y >= n || z < 0 || z >= n) {
			// starting point outside
			fprintf(stderr, "starting point outside grid!\n");
			fprintf(stderr, "ray       : %f %f %f | %f %f %f | %f\n", x0, y0, z0, dx, dy, dz, d);
			fprintf(stderr, "step sizes: %f %f %f\n", t_next_x, t_next_y, t_next_z);
			assert(0);
		}
		while(1) {
			IFDEBUG printf("step sizes: %f %f %f\n", t_next_x, t_next_y, t_next_z);
			IFDEBUG printf("step sizes: %f %f %f (with signs)\n", 
				t_next_x * sign_x, t_next_y * sign_y, t_next_z * sign_z);
			if (t_next_y * sign_y <= min(t_next_x * sign_x, t_next_z * sign_z)) {
				y += y_inc;
				t = t_next_y;
				t_next_y += dt_dy;
				IFDEBUG printf("     go in y %f\n", t);
			} else if (t_next_x * sign_x <= min(t_next_y * sign_y, t_next_z * sign_z)) {
				x += x_inc;
				t = t_next_x;
				t_next_x += dt_dx;
				IFDEBUG printf("     go in x %f\n", t);
			} else {
				z += z_inc;
				t = t_next_z;
				t_next_z += dt_dz;
				IFDEBUG printf("     go in z %f\n", t);
			}
			double rho = rhoarr[last_x * n * n + last_y * n + last_z];
			if ((fabs(t) - last_t) * rho > d) {
				// compute fraction in current cell
				IFDEBUG printf("     found endpoint\n");
				tout[i] = last_t + d / rho;
				break;
			}
			// subtract full cell
			IFDEBUG printf("     passing cell %d %d %d : d=%f - %f x %f\n", last_x, last_y, last_z, d, fabs(t) - last_t, rho);
			d -= (fabs(t) - last_t) * rho;
			// check if we left
			if (x < 0 || x >= n || y < 0 || y >= n || z < 0 || z >= n) {
				// we left the grid
				IFDEBUG printf("     went to inf\n");
				tout[i] = -1;
				break;
			}
			last_t = fabs(t);
			last_x = x;
			last_y = y;
			last_z = z;
		}
	}
	return 0;
}


/**
 * ray tracing on a grid
 *
 * Parameters regarding the uniform, cartesian grid:
 * rho:    double array: density for conversion from length to column density
 * n:      length of rho
 * Parameters regarding the integration direction:
 * x:      double array: position vector
 * y:      double array: position vector
 * z:      double array: position vector
 * a:      double array: direction vector
 * b:      double array: direction vector
 * c:      double array: direction vector
 * m:      length of a, b, c
 * Output:
 * NHout   double array: NH integrated to infinity
 */
static int grid_raytrace(
	const double * rhop, int n, 
	const double * xp, const double * yp, const double * zp, const double * ap, const double * bp, const double * cp, int m, 
	const double * NHoutp
) {
	double * rhoarr = (double*) rhop;
	double * NHout = (double*) NHoutp;
	
	#ifdef PARALLEL
	#pragma omp parallel for
	#endif
	for (int i = 0; i < m; i++) { // one ray at a time
		const double dx = ((double*) ap)[i];
		const double dy = ((double*) bp)[i];
		const double dz = ((double*) cp)[i];
		const double x0 = ((double*) xp)[i];
		const double y0 = ((double*) yp)[i];
		const double z0 = ((double*) zp)[i];
		
		const double dt_dx = 1. / dx;
		const double dt_dy = 1. / dy;
		const double dt_dz = 1. / dz;
		
		int x = x0;
		int y = y0;
		int z = z0;
		int x_inc, y_inc, z_inc;
		double t_next_x, t_next_y, t_next_z;
		int sign_x = 1, sign_y = 1, sign_z = 1;
		
		if (dx == 0) {
			x_inc = 0;
			t_next_x = dt_dx; // inf
		} else if (dx > 0) {
			x_inc = 1;
			t_next_x = (floor(x0) + 1 - x0) / dx;
		} else {
			x_inc = -1;
			t_next_x = (x0 - floor(x0)) / dx;
			sign_x = -1;
		}
		if (dy == 0) {
			y_inc = 0;
			t_next_y = dt_dy; // inf
		} else if (dy > 0) {
			y_inc = 1;
			t_next_y = (floor(y0) + 1 - y0) / dy;
		} else {
			y_inc = -1;
			t_next_y = (y0 - floor(y0)) / dy;
			sign_y = -1;
		}
		if (dz == 0) {
			z_inc = 0;
			t_next_z = dt_dz; // inf
		} else if (dz > 0) {
			z_inc = 1;
			t_next_z = (floor(z0) + 1 - z0) / dz;
		} else {
			z_inc = -1;
			t_next_z = (z0 - floor(z0)) / dz;
			sign_z = -1;
		}
		IFDEBUG printf("ray       : %f %f %f | %f %f %f\n", x0, y0, z0, dx, dy, dz);
		IFDEBUG printf("step sizes: %f %f %f\n", t_next_x, t_next_y, t_next_z);
		double t;
		double last_t = 0;
		int last_x = x;
		int last_y = y;
		int last_z = z;
		if (x < 0 || x >= n || y < 0 || y >= n || z < 0 || z >= n) {
			// starting point outside
			assert(0);
		}
		NHout[i] = 0;
		while(1) {
			IFDEBUG printf("step sizes: %f %f %f\n", t_next_x, t_next_y, t_next_z);
			if (t_next_y * sign_y <= min(t_next_x * sign_x, t_next_z * sign_z)) {
				y += y_inc;
				t = t_next_y;
				t_next_y += dt_dy;
				IFDEBUG printf("     go in y %f\n", t);
			} else if (t_next_x * sign_x <= min(t_next_y * sign_y, t_next_z * sign_z)) {
				x += x_inc;
				t = t_next_x;
				t_next_x += dt_dx;
				IFDEBUG printf("     go in x %f\n", t);
			} else {
				z += z_inc;
				t = t_next_z;
				t_next_z += dt_dz;
				IFDEBUG printf("     go in z %f\n", t);
			}
			double rho = rhoarr[last_x * n * n + last_y * n + last_z];
			NHout[i] += (fabs(t) - last_t) * rho;
			IFDEBUG printf("     NHout=%f (%f...%f, rho=%.1f)\n", NHout[i], fabs(t), last_t, rho);
			// check if we left
			if (x < 0 || x >= n || y < 0 || y >= n || z < 0 || z >= n) {
				// we left the grid
				IFDEBUG printf("     went to inf\n");
				break;
			}
			// subtract full cell
			last_t = fabs(t);
			last_x = x;
			last_y = y;
			last_z = z;
		}
	}
	return 0;
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
	const double * xx, const double * yy, const double * zz, const double * RR, const int n
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
 * Segmentation of sphere with cone cut-outs
 *
 * a, b, c: direction vector
 * x, y, z: position
 * 
 * cones:
 * thetas: opening angles, ordered with lowest value first
 * n:      number of cones
 * 
 * returns a segmentation of the line with xlo, xhi values in order.
 */
struct segmentation cone_segments_create(
	const double a, const double b, const double c, 
	const double x, const double y, const double z, 
	const double * thetas, const int n
	) {
	// rdir should be 1, but... you never know
	double rdir = sqrt(pow(a, 2) + pow(b, 2) + pow(c, 2));
	double beta = (rdir == 0 ? 0 : acos(c / rdir));
	int i, j;
	
	// trivial case first
	if (x == 0 && y == 0 && z == 0) {
		IFDEBUG fprintf(stderr, "trivial case, from origin\n");
		// simplify our lives by z-symmetry
		IFDEBUG fprintf(stderr, "  beta is %.3f\n", beta);
		if (beta > M_PI_2) {
			beta = M_PI - beta;
			IFDEBUG fprintf(stderr, "  beta mirrored -> is %.3f\n", beta);
		}
		// if too steep angle, will not hit anything, no matter there
		if (beta < thetas[0]) {
			IFDEBUG fprintf(stderr, "  trivial case, to pole\n");
			struct segmentation s;
			s.n = 0;
			s.segments = NULL;
			return s;
		}
		
		// there is a single positive intersection with the sphere
		// at radius 1.
		struct segmentation s;
		s.n = 1;
		s.segments = (struct segment *) calloc(1, sizeof(struct segment));
		assert(s.segments != 0);
		
		// the density is from the cone that brackets the direction
		for (i = 0; i < n; i++) {
			// mark with last segment that covers this ray
			if (beta > thetas[i]) {
				IFDEBUG fprintf(stderr, "  trivial case, segment could be %d\n", i);
				struct segment seg = {0, 1, i};
				s.segments[0] = seg;
			}
		}
		return s;
	}
	
	// compute position in spherical coordinates
	double r = sqrt(pow(a, 2) + pow(b, 2) + pow(c, 2));
	double theta = (r == 0 ? 0 : acos(z / r));
	IFDEBUG fprintf(stderr, "  theta is %.3f\n", theta);
	if (theta > M_PI_2) {
		theta = M_PI - theta;
		IFDEBUG fprintf(stderr, "  theta mirrored -> is %.3f\n", beta);
	}

	struct segmentation s;
	s.n = n;
	s.segments = (struct segment *) calloc(s.n, sizeof(struct segment));
	assert(s.segments != 0);
	double edges[2*n];
	// for each cone create a intersection segment
	for (i = 0; i < n; i++) {
		// we have to pass the origin of the cone, which is shifted
		// compared to our location.
		struct segment seg = single_cone_intersection(a, b, c, x, y, z, thetas[i]);
		s.segments[i] = seg;
		edges[i*2] = seg.xlo;
		edges[i*2+1] = seg.xhi;
		seg.i = i;
	}
	
	// create merged segmentation from edge points
	for (i = 0; i < 2*n; i++) {
		IFDEBUG fprintf(stderr, "   edge %d: %.3f\n", i, edges[i]);
	}
	qsort(edges, 2*n, sizeof(double), cmp_double);
	for (i = 0; i < 2*n; i++) {
		IFDEBUG fprintf(stderr, "   edge %d: %.3f\n", i, edges[i]);
	}
	struct segmentation smerged;
	smerged.n = 2*n + 1;
	smerged.segments = (struct segment *) calloc(smerged.n, sizeof(struct segment));
	assert(smerged.segments != 0);
	
	IFDEBUG fprintf(stderr, "all edges: %d\n", smerged.n);
	smerged.segments[0].xlo = -1e100;
	smerged.segments[0].xhi = edges[0];
	smerged.segments[0].i = -1;
	IFDEBUG fprintf(stderr, "   -inf\n");
	IFDEBUG fprintf(stderr, "   %.3f\n", edges[0]);
	for (i = 1; i < 2*n; i++) {
		smerged.segments[i].xlo = smerged.segments[i-1].xhi;
		smerged.segments[i].xhi = edges[i];
		IFDEBUG fprintf(stderr, "   %.3f\n", edges[i]);
		smerged.segments[i].i = -1; // do not know at the moment
	}
	smerged.segments[2*n].xlo = smerged.segments[2*n-1].xhi;
	smerged.segments[2*n].xhi = 1e100;
	IFDEBUG fprintf(stderr, "   +inf\n");
	smerged.segments[2*n].i = -1;
	for (j = 0; j < smerged.n; j++) {
		IFDEBUG fprintf(stderr, "   segment %d: [%.3f - %.3f]\n", j, smerged.segments[j].xlo, smerged.segments[j].xhi);
	}
	
	// mark each segment with the correct cone index
	for (i = 0; i < s.n; i++) {
		// go through each cone in turn to color segments 
		// --> last cone color prevails
		
		double xlo = s.segments[i].xlo;
		double xhi = s.segments[i].xhi;
		if (xlo > xhi) {
			double xtmp = xlo;
			xlo = xhi;
			xhi = xtmp;
		}
		IFDEBUG fprintf(stderr, "marking cone %d, had cut segment [%.3f-%.3f]: (we are %s)\n", 
			i, xlo, xhi, (theta > thetas[i]) ? "inside" : "outside");
		int inverted = 0;
		// if we start inside the cone but outside the cone cut segment
		// e.g. cutting horizontally from outside, or from the top
		if (theta > thetas[i] && (xlo > 0 || xhi < 0)) {
			// have to mark everything but the cone cut segment
			inverted = 1;
		}
		// if we start outside the cone but inside the cone cut segment
		// e.g. above the origin
		if (theta < thetas[i] && (xlo <= 0 && xhi >= 0)) {
			// have to mark everything but the cone cut segment
			inverted = 1;
		}
		
		for (j = 0; j < smerged.n; j++) {
			// go through each segment
			double ylo = smerged.segments[j].xlo;
			double yhi = smerged.segments[j].xhi;
			IFDEBUG fprintf(stderr, "  segment %d %.3f-%.3f\n", j, ylo, yhi);
			if (ylo >= xlo && yhi <= xhi) {
				// inside the cone cut
				// mark unless inverted
				if (inverted) {
					IFDEBUG fprintf(stderr, "    overlapping, not marking\n");
					continue;
				}
			} else {
				// we are outside this cone
				// mark if inverted
				if (!inverted) {
					IFDEBUG fprintf(stderr, "    not overlapping, not marking\n");
					continue;
				}
			}
			// mark now
			// because of the i-loop being outside, the highest
			// density cone gets marked last (if they are ordered
			// by density).
			smerged.segments[j].i = i;
			IFDEBUG fprintf(stderr, "    -> marking as cone %d\n", i);
		}
	}
	
	// discard segments using sphere intersection
	// passing the location of the sphere as argument, which is shifted from 
	// the origin by the location of the line origin
	struct segment seg = single_sphere_intersection(a, b, c, 0-x, 0-y, 0-z, 1);
	double xlo = seg.xlo;
	double xhi = seg.xhi;
	if (xlo > xhi) {
		double xtmp = xlo;
		xlo = xhi;
		xhi = xtmp;
	}
	IFDEBUG fprintf(stderr, "sphere intersection [%.3f,%.3f,%.3f] -> [%.3f,%.3f,%.3f]: [%.3f-%.3f]\n", x, y, z, a, b, c, xlo, xhi);
	
	for (j = 0; j < smerged.n; j++) {
		double ylo = smerged.segments[j].xlo;
		double yhi = smerged.segments[j].xhi;
		IFDEBUG fprintf(stderr, "  segment %d [%.3f-%.3f]\n", j, ylo, yhi);
		if (yhi <= xlo || ylo >= xhi) {
			// outside the sphere, mark as empty
			smerged.segments[j].i = -1;
			IFDEBUG fprintf(stderr, "    completely outside sphere, marking as empty\n");
			continue;
		}
		if (ylo < xlo && yhi > xlo) {
			// entering sphere, left side is outside.
			smerged.segments[j].xlo = xlo;
			IFDEBUG fprintf(stderr, "    partially (left) outside sphere, cropping\n");
		}
		if (ylo < xhi && yhi > xhi) {
			// leaving sphere, right side is outside.
			smerged.segments[j].xhi = xhi;
			IFDEBUG fprintf(stderr, "    partially (right) outside sphere, cropping\n");
		}
		IFDEBUG {
			ylo = smerged.segments[j].xlo;
			yhi = smerged.segments[j].xhi;
			fprintf(stderr, "     -> [%.3f-%.3f]\n", ylo, yhi);
		}
	}
	
	// count how many are left
	IFDEBUG fprintf(stderr, "cleaning up ...\n");
	int m = 0;
	for (j = 0; j < smerged.n; j++) {
		if (smerged.segments[j].i != -1 && smerged.segments[j].xlo != smerged.segments[j].xhi) {
			m++;
		}
	}
	
	// create a stripped version of smerged and return it.
	IFDEBUG fprintf(stderr, "only have %d segments (instead of %d):\n", m, smerged.n);
	struct segmentation scropped;
	scropped.n = m;
	scropped.segments = (struct segment *) calloc(m, sizeof(struct segment));
	assert(scropped.segments != 0);
	
	i = 0;
	for (j = 0; j < smerged.n; j++) {
		if (smerged.segments[j].i != -1 && smerged.segments[j].xlo != smerged.segments[j].xhi) {
			struct segment seg;
			seg = smerged.segments[j];
			IFDEBUG fprintf(stderr, "  segment %d [%.3f-%.3f] from cone %d\n", j, seg.xlo, seg.xhi, seg.i);
			scropped.segments[i] = seg;
			i++;
		}
	}
	IFDEBUG fprintf(stderr, "  freeing smerged segments\n");
	free(smerged.segments);
	IFDEBUG fprintf(stderr, "  freeing s segments\n");
	free(s.segments);
	IFDEBUG fprintf(stderr, "  segmentation done\n");
	return scropped;
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
static int voronoi_raytrace(
	const double * xxp, const double * yyp, const double * zzp, const double * RRp, const double * rhop, int n, 
	const double * ap, const double * bp, const double * cp, int m, 
	const double * mindistancesp, int l, const double * NHoutp
) {
	const double * mindistances = (double*) mindistancesp;
	double * NHout = (double*) NHoutp;
	
	#ifdef PARALLEL
	#pragma omp parallel for
	#endif
	for (int i = 0; i < m; i++) { // one ray at a time
		IFDEBUG fprintf(stderr, "ray %d/%d\n", i, m);
		fprintf(stderr, "ray %d/%d through %d    \r", i, m, n);
		
		const double a = ((double*) ap)[i];
		const double b = ((double*) bp)[i];
		const double c = ((double*) cp)[i];
		
		
		double NHtotal[l];
		for (int k = 0; k < l; k++) {
			NHtotal[k] = 0.0;
		}
		const double * xx = ((double*) xxp);
		const double * yy = ((double*) yyp);
		const double * zz = ((double*) zzp);
		const double * RR = ((double*) RRp);
		const double * rho = ((double*) rhop);
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
 * d:      double array: distance to travel
 * x:      double array: coordinate vector
 * y:      double array: coordinate vector
 * z:      double array: coordinate vector
 * m:      length of a, b, c, x, y, z
 */
static int voronoi_raytrace_finite(
	const double * xxp, const double * yyp, const double * zzp, const double * RRp, const double * rhop, int n, 
	const double * ap, const double * bp, const double * cp, const double * dp, 
	const double * xp, const double * yp, const double * zp, int m
) {
	#ifdef PARALLEL
	#pragma omp parallel for
	#endif
	for (int i = 0; i < m; i++) { // one ray at a time
		IFDEBUG fprintf(stderr, "ray %d/%d\n", i, m);
		fprintf(stderr, "ray %d/%d through %d    \r", i, m, n);
		
		const double a = ((double*) ap)[i];
		const double b = ((double*) bp)[i];
		const double c = ((double*) cp)[i];
		const double d = ((double*) dp)[i];

		const double x = ((double*) xp)[i];
		const double y = ((double*) yp)[i];
		const double z = ((double*) zp)[i];
		
		const double * xx = ((double*) xxp);
		const double * yy = ((double*) yyp);
		const double * zz = ((double*) zzp);
		const double * RR = ((double*) RRp);
		const double * rho = ((double*) rhop);
		IFVERBOSE fprintf(stderr, "  segmenting...\n");
		struct segmentation s = segments_create(a, b, c, x, y, z,
			xx, yy, zz, RR, n);
		IFVERBOSE fprintf(stderr, "  segmenting done.\n");
		// now go through the segmentation and sum from mindistances[l] up.
		double dcum = 0;
		double t = -1;
		for (int j = 0; j < s.n; j++) { // one segment at a time
			struct segment seg = s.segments[j];
			double xlo = seg.xlo;
			double xhi = seg.xhi;
			double density = rho[seg.i];
			IFDEBUG fprintf(stderr, "  segment %d/%d: %.3f %.3f %.3e \n", j, s.n, xlo, xhi, density);
			if (xhi < 0)
				continue;
			if (xlo < 0)
				xlo = 0;
			
			double dseg = density * (xhi - xlo);
			if (dseg + dcum > d) {
				// we ran out of distance to travel, so we 
				// should stop inside this segment.
				
				// how far along should we go?
				t = ((d - dcum) / dseg) * (xhi - xlo) + xlo;
				break;
			} else {
				// we are not done yet, go through this segment
				dcum += dseg;
			}
		}
		free(s.segments);
		if (t == -1) {
			// we stepped outside
			t = 1e10;
		}
		// lets figure out the corresponding coordinates
		((double * )xp)[i] = x + a*t;
		((double * )yp)[i] = y + b*t;
		((double * )zp)[i] = z + c*t;
	}
	return 0;
}


/**
 * ray tracing through a sphere/cone cuts
 * 
 * Sphere radius is 1, each cone angle defines a region of a certain density.
 * 
 * This function raytraces from the starting coordinates in the direction
 * given and compute the column density of the intersecting segments.
 * Then it will go along the ray in the positive direction until the
 * column density d is reached. The coordinates of that point are stored into 
 * (x, y, z)
 *
 * Parameters regarding the cones:
 * thetas: double array: cone opening angle
 * rhos:   double array: density of each cone from length to column density
 * n:      number of cones
 * Parameters regarding the integration direction:
 * x:      double array: coordinates
 * y:      double array: coordinates
 * z:      double array: coordinates
 * a:      double array: direction vector
 * b:      double array: direction vector
 * c:      double array: direction vector
 * m:      length of (a, b, c) and (x,y,z)
 * d:      double array: distance to travel
 * Output:
 * t       double array: end position along direction vector. -1 if infinite
 */
static int cone_raytrace_finite(
	const double * thetasp, const double * rhosp, int n, 
	double * xp, double * yp, double * zp, 
	const double * ap, const double * bp, const double * cp, int m,
	const double * dp, const double * tp
) {
	const double * thetas = (double*) thetasp;
	const double * rhos = (double*) rhosp;
	
	#ifdef PARALLEL
	#pragma omp parallel for
	#endif
	for (int i = 0; i < m; i++) { // one ray at a time
		IFDEBUG fprintf(stderr, "ray %d/%d\n", i, m);
		//fprintf(stderr, "ray %d/%d through %d    \r", i, m, n);
		
		const double a = ((double*) ap)[i];
		const double b = ((double*) bp)[i];
		const double c = ((double*) cp)[i];
		
		const double x = ((double*) xp)[i];
		const double y = ((double*) yp)[i];
		const double z = ((double*) zp)[i];

		const double d = ((double*) dp)[i];
		
		IFVERBOSE fprintf(stderr, "  segmenting...\n");
		struct segmentation s = cone_segments_create(a, b, c, 
			x, y, z, thetas, n);
		IFVERBOSE fprintf(stderr, "  segmenting done.\n");
		// now go through the segmentation and sum from mindistances[l] up.
		double dcum = 0;
		double t = -1;
		for (int j = 0; j < s.n; j++) { // one segment at a time
			struct segment seg = s.segments[j];
			double xlo = seg.xlo;
			double xhi = seg.xhi;
			double density = rhos[seg.i];
			IFDEBUG fprintf(stderr, "  segment %d/%d: %.3f %.3f %.3e \n", j, s.n, xlo, xhi, density);
			if (xhi < 0)
				continue;
			if (xlo < 0)
				xlo = 0;
			double dseg = density * (xhi - xlo);
			if (dseg + dcum > d) {
				// we ran out of distance to travel, so we 
				// should stop inside this segment.
				
				// how far along should we go?
				t = ((d - dcum) / dseg) * (xhi - xlo) + xlo;
				IFDEBUG fprintf(stderr, "    stopping inside at %.3f \n", t);
				break;
			} else {
				// we are not done yet, go through this segment
				dcum += dseg;
				IFDEBUG fprintf(stderr, "    going through: travelled d=%.3f (of %.3f planned)\n", dcum, d);
			}
		}
		free(s.segments);
		IFDEBUG fprintf(stderr, "  ended up at t=%.3f\n", t);
		((double*) tp)[i] = t;
	}
	return 0;
}


/**
 * marks the first kstop non-intersecting spheres
 *
 * Parameters regarding the spheres:
 * xx:     double array: coordinates
 * yy:     double array: coordinates
 * zz:     double array: coordinates
 * RR:     double array: sphere radius
 * n:      int: length of xx,yy,zz,RR
 * kstop:  int: number of spheres desired
 * NHout   double array: output; of size n
 * 
 */
static int sphere_sphere_collisions(
	const double * xxp, const double * yyp, const double * zzp, const double * RRp, int n, int kstop, 
	const double * NHoutp
) {
	double * NHout = (double*) NHoutp;
	int i = 0;
	for (; i < n; i++) { // one sphere at a time
		const double a = ((double*) xxp)[i];
		const double b = ((double*) yyp)[i];
		const double c = ((double*) zzp)[i];
		const double R = ((double*) RRp)[i];
		int collides = 0;
		
		#ifdef PARALLEL
		#pragma omp parallel for shared(collides)
		#endif
		for (int j = 0; j < i; j++) { // one sphere at a time
			if (NHout[j] == 1 || collides != 0) {
				// only look at those not already marked bad
				continue;
			}
			const double xx = ((double*) xxp)[j];
			const double yy = ((double*) yyp)[j];
			const double zz = ((double*) zzp)[j];
			const double RR = ((double*) RRp)[j];
			const double dist = pow(pow(a - xx, 2) + pow(b - yy, 2) + pow(c - zz, 2), 0.5);
			if (dist < RR + R) {
				collides = 1; // mark bad
			}
		}
		NHout[i] = collides;
		if (collides == 0) {
			kstop--;
			if (kstop < 0) 
				break;
			if (kstop % 100 == 0)
				fprintf(stderr, "todo: %d \r", kstop);
		}
	}
	for (; i < n; i++) {
		NHout[i] = 2;
	}
	fprintf(stderr, "done        \n");
	return 0;
}

