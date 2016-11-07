LightRayRider
========================================================
 
A fast library for calculating intersections of a line with many spheres or inhomogeneous material.

Introduction
-------------

In astrophysics, a point source can be obscured by a gas distribution along the line-of-sight.
For hydrodynamic simulations which produce such a gas distribution, this code can compute
the total density along a arbitrary ray. The output is a column density, 
also known as N_H if hydrogen gas is irradiated.

Line/Sphere cutting
--------------------

Input:

* Points in space representing sphere centres.
* Sphere radii and densities.
* One or more arbitrary lines from the origin.

Output:

* This computes the total length / column density cut.
* From distance 0 or another chosen minimal distance (or multiple).

Method:

* Simple quadratic equations.

Voronoi cutting
----------------------

Input:

* Points in space. 
* Densities.
* One or more arbitrary lines from the origin

Output:

* This computes the total length along the line,
  where every point on the line is assigned the density from the 
  nearest point (Voronoi segmentation).
* From distance 0 or another chosen minimal distance (or multiple).

Method:

* Segmentation of the line where points become equi-distant. 
  Performs approximately linearly with number of points.

Usage
--------------

Compile the c library first with::

	$ make 

To use from Python, use raytrace.py::
	
	from raytrace import voronoi_raytrace, sphere_raytrace

You can find the declaration of how to call these functions in raytrace.py.
Basically, you pass the coordinates of your gas particles, the associated
densities and the starting point and direction of your raytracing.

Example usage is demonstrated in irradiate.py. This was used for Illustris and 
EAGLE particle in the associated paper.

Parallel processing
-----------------------

LightRayRider supports multiple processors through OpenMP.
Set the variable OMP_NUM_THREADS to the number of processors you want to use,
and the parallel library ray-parallel.so will be loaded.

License and Acknowledgements
--------------------------------

If you use this code, please cite "Galaxy gas as obscurer: II. Separating the galaxy-scale and
nuclear obscurers of Active Galactic Nuclei", by Buchner & Bauer (2016, submitted).
https://arxiv.org/abs/1610.09380

I will put a bibcode here as soon as the paper is published.

The code is licensed under AGPLv3 (see LICENSE file).



