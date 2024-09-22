"""
LightRayRider is a tiny fast library to compute column densities along a ray.

Supported geometries:

* Uniform density grids.
* Spheres of varying size, density and position.
* Co-centred cones of varying opening angles and density.
"""
import os

__version__ = '2.0.6'

print(
    """
You are using the LightRayRider library, which provides optimized calls for
photon propagation and column density computations.
Please cite: Buchner & Bauer (2017), """
    "http://adsabs.harvard.edu/abs/2017MNRAS.465.4348B")

if int(os.environ.get('OMP_NUM_THREADS', '1')) > 1:
    from . import parallel as raytrace
    print("Parallelisation enabled.\n")
else:
    from . import raytrace
    print("Parallelisation disabled (use OMP_NUM_THREADS to enable).\n")

sphere_raytrace = raytrace.py_sphere_raytrace
sphere_raytrace_count_between = raytrace.py_sphere_raytrace_count_between
grid_raytrace = raytrace.py_grid_raytrace
grid_raytrace_flat = raytrace.py_grid_raytrace_flat
voronoi_raytrace = raytrace.py_voronoi_raytrace
sphere_sphere_collisions = raytrace.py_sphere_sphere_collisions
sphere_raytrace_finite = raytrace.py_sphere_raytrace_finite
cone_raytrace_finite = raytrace.py_cone_raytrace_finite
grid_raytrace_finite = raytrace.py_grid_raytrace_finite
grid_raytrace_finite_flat = raytrace.py_grid_raytrace_finite_flat
