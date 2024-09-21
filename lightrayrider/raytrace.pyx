# cython: language_level=3,annotate=True,profile=True,fast_fail=True,warning_errors=True
"""
This file is part of LightRayRider, a fast column density computation tool.
"""
import numpy as np
cimport numpy as cnp
cnp.import_array()
cimport cython

cdef extern from "ray.c":
    int sphere_raytrace(
        const double * xxp, const double * yyp, const double * zzp, const double * RRp, const double * rhop, int n,
        const double * ap, const double * bp, const double * cp, int m,
        const double * mindistancesp, int l, const double * NHoutp
    )
    int sphere_raytrace_count_between(
        const double * xxp, const double * yyp, const double * zzp, const double * RRp, int n,
        const double * ap, const double * bp, const double * cp, int m,
        const double * NHoutp
    )
    int sphere_raytrace_finite(
        const double * xxp, const double * yyp, const double * zzp, const double * RRp, const double * rhop, int n,
        const double * xp, const double * yp, const double * zp, const double * ap, const double * bp, const double * cp, int m,
        const double * NHmaxp, const double * tp
    )
    int grid_raytrace_finite(
        const double * rhop, int n,
        const double * xp, const double * yp, const double * zp, const double * ap, const double * bp, const double * cp, int m,
        const double * NHmaxp, const double * tp
    )
    int grid_raytrace(
        const double * rhop, int n,
        const double * xp, const double * yp, const double * zp, const double * ap, const double * bp, const double * cp, int m,
        const double * NHoutp
    )
    int voronoi_raytrace(
        const double * xxp, const double * yyp, const double * zzp, const double * RRp, const double * rhop, int n,
        const double * ap, const double * bp, const double * cp, int m,
        const double * mindistancesp, int l, const double * NHoutp
    )
    int voronoi_raytrace_finite(
        const double * xxp, const double * yyp, const double * zzp, const double * RRp, const double * rhop, int n,
        const double * ap, const double * bp, const double * cp, const double * dp,
        const double * xp, const double * yp, const double * zp, int m
    )
    int cone_raytrace_finite(
        const double * thetasp, const double * rhosp, int n,
        double * xp, double * yp, double * zp,
        const double * ap, const double * bp, const double * cp, int m,
        const double * dp, const double * tp
    )
    int sphere_sphere_collisions(
        const double * xxp, const double * yyp, const double * zzp, const double * RRp, int n, int kstop,
        const double * NHoutp
    )


@cython.boundscheck(False)
@cython.wraparound(False)
def py_sphere_raytrace(
    double[::1] xx,
    double[::1] yy,
    double[::1] zz,
    double[::1] RR,
    double[::1] rho,
    double[::1] a,
    double[::1] b,
    double[::1] c,
    double[::1] mindistances
):
    """
    Perform ray tracing using sphere intersections.

    Parameters
    ----------
    xx: array
        sphere x coordinates
    yy: array
        sphere y coordinates
    zz: array
        sphere z coordinates
    RR: array
        sphere radii
    rho: array
        sphere densities (in units of 1e22/cm^2). For conversion from length to column density.
    a: array
        ray direction, x component
    b: array
        ray direction, y component
    c: array
        ray direction, z component
    mindistances: array
        distance from the origin for each subtotal to consider.

    Returns
    ----------
    NHout: array
        column density from the origin to infinity. shape is (len(mindistances), len(a)).
    """
    cdef int lenxx = xx.shape[0]
    cdef int lena = a.shape[0]
    cdef int lenmd = mindistances.shape[0]
    cdef cnp.ndarray[double, ndim=1] NHout = np.zeros(lena * lenmd) - 1

    cdef int r = sphere_raytrace(
        &xx[0], &yy[0], &zz[0], &RR[0], &rho[0], lenxx,
        &a[0], &b[0], &c[0], lena, &mindistances[0], lenmd, &NHout[0])
    if r != 0:
        raise RuntimeError("sphere_raytrace calculation failed")
    return NHout.reshape((lenmd, lena))


@cython.boundscheck(False)
@cython.wraparound(False)
def py_sphere_raytrace_count_between(
    double[::1] xx,
    double[::1] yy,
    double[::1] zz,
    double[::1] RR,
    double[::1] a,
    double[::1] b,
    double[::1] c
):
    """
    Count number of spheres lying (partially or completely) between point a,b,c and origin.

    Stops counting at 1.

    Parameters
    ----------
    xx: array
        sphere x coordinates
    yy: array
        sphere y coordinates
    zz: array
        sphere z coordinates
    RR: array
        sphere radii
    a: array
        ray direction, x component
    b: array
        ray direction, y component
    c: array
        ray direction, z component

    Returns
    ----------
    n: int
        number of spheres
    """
    cdef int lenxx = xx.shape[0]
    cdef int lena = a.shape[0]

    # Allocate output array
    cdef cnp.ndarray[double, ndim=1] NHout = np.zeros(lena) - 1

    # Call the external C function
    return sphere_raytrace_count_between(
        &xx[0], &yy[0], &zz[0], &RR[0], lenxx, &a[0], &b[0], &c[0], lena, &NHout[0])


@cython.boundscheck(False)
@cython.wraparound(False)
def py_grid_raytrace(
    cnp.ndarray[double, ndim=3, mode="c"] rho,
    double[::1] x,
    double[::1] y,
    double[::1] z,
    double[::1] a,
    double[::1] b,
    double[::1] c
):
    """
    Ray tracing on a grid.

    Parameters
    ----------
    rho: array
        3D cube of densities (in units of 1e22/cm^2). For conversion from length to column density.
        Must be same size in each axis.
    lenrho: int
        length of rho.
    x: array
        ray start coordinate, x component
    y: array
        ray start coordinate, y component
    z: array
        ray start coordinate, z component
    a: array
        ray direction, x component
    b: array
        ray direction, y component
    c: array
        ray direction, z component

    Returns
    -------
    NHout: array
        column density from the origin to infinity. shape is len(a).
    """
    cdef int lenrho = rho.shape[0]
    assert lenrho == rho.shape[1]
    assert lenrho == rho.shape[2]
    cdef const double[::1] rho_flat = np.array(rho.flatten())
    return py_grid_raytrace_flat(np.array(rho.flatten()), lenrho, x, y, z, a, b, c)


@cython.boundscheck(False)
@cython.wraparound(False)
def py_grid_raytrace_flat(
    double[::1] rho_flat,
    int lenrho,
    double[::1] x,
    double[::1] y,
    double[::1] z,
    double[::1] a,
    double[::1] b,
    double[::1] c
):
    """
    Ray tracing on a grid.

    Parameters
    ----------
    rho_flat: array
        flattened 3D cube of densities. For conversion from length to column density.
    lenrho: int
        length of the original rho cube in each axis.
    x: array
        ray start coordinate, x component
    y: array
        ray start coordinate, y component
    z: array
        ray start coordinate, z component
    a: array
        ray direction, x component
    b: array
        ray direction, y component
    c: array
        ray direction, z component

    Returns
    -------
    NHout: array
        column density from the origin to infinity. shape is len(a).
    """
    cdef int lena = len(a)
    cdef cnp.ndarray[double, ndim=1] NHout = np.zeros(lena) - 1
    cdef int r = grid_raytrace(
        &rho_flat[0], lenrho, &x[0], &y[0], &z[0], &a[0], &b[0], &c[0], lena, &NHout[0])
    if r != 0:
        raise RuntimeError("grid_raytrace calculation failed")
    return NHout


@cython.boundscheck(False)
@cython.wraparound(False)
def py_voronoi_raytrace(
    double[::1] xx,
    double[::1] yy,
    double[::1] zz,
    double[::1] RR,
    double[::1] rho,
    double[::1] a,
    double[::1] b,
    double[::1] c,
    double[::1] mindistances
):
    """
    Ray tracing using nearest point density.

    Parameters
    ----------
    xx: array
        x coordinates
    yy: array
        y coordinates
    zz: array
        z coordinates
    RR: array
        radii
    rho: array
        sphere densities. For conversion from length to column density.
    a: array
        ray direction, x component
    b: array
        ray direction, y component
    c: array
        ray direction, z component
    mindistances: array
        distance from the origin for each subtotal to consider.

    Returns
    -------
    NHout: array
        column density from the origin to infinity. shape is (len(mindistances), len(a)).
    """
    cdef int lenxx = len(xx)
    cdef int lena = len(a)
    cdef int lenmd = len(mindistances)
    cdef cnp.ndarray[double, ndim=1] NHout = np.zeros(lena * lenmd) - 1
    assert xx.shape[0] == lenxx, xx.shape[0]
    assert yy.shape[0] == lenxx, yy.shape[0]
    assert zz.shape[0] == lenxx, zz.shape[0]
    assert RR.shape[0] == lenxx, RR.shape[0]
    assert rho.shape[0] == lenxx, rho.shape[0]
    assert a.shape[0] == lena, a.shape[0]
    assert b.shape[0] == lena, b.shape[0]
    assert c.shape[0] == lena, c.shape[0]
    assert mindistances.shape[0] == lenmd, mindistances.shape[0]
    assert NHout.shape[0] == lena * lenmd, NHout.shape[0]
    cdef int r = voronoi_raytrace(
        &xx[0], &yy[0], &zz[0], &RR[0], &rho[0], lenxx, &a[0], &b[0], &c[0], lena, &mindistances[0], lenmd, &NHout[0])
    if r != 0:
        raise RuntimeError("voronoi_raytrace calculation failed")
    return NHout.reshape((len(mindistances), -1))


@cython.boundscheck(False)
@cython.wraparound(False)
def py_sphere_sphere_collisions(double[::1] xx,
                                double[::1] yy,
                                double[::1] zz,
                                double[::1] RR,
                                int kstop):
    """
    Marks the first kstop non-intersecting spheres.

    Parameters
    ----------
    xx: array
        sphere x coordinates
    yy: array
        sphere y coordinates
    zz: array
        sphere z coordinates
    RR: array
        sphere radii
    k: int
        number of spheres desired

    Returns
    ----------
    NHout: array
        1 if collides with another sphere of lower index,
        2 if okay.
    Parameters regarding the spheres:
    """
    cdef int lenxx = len(xx)
    cdef cnp.ndarray[double, ndim=1] NHout = np.zeros(lenxx) - 1
    cdef int r = sphere_sphere_collisions(
        &xx[0], &yy[0], &zz[0], &RR[0], lenxx, kstop, &NHout[0])
    if r != 0:
        raise RuntimeError("sphere_sphere_collisions calculation failed")
    return NHout


@cython.boundscheck(False)
@cython.wraparound(False)
def py_sphere_raytrace_finite(
    double[::1] xx,
    double[::1] yy,
    double[::1] zz,
    double[::1] RR,
    double[::1] rho,
    double[::1] x,
    double[::1] y,
    double[::1] z,
    double[::1] a,
    double[::1] b,
    double[::1] c,
    double[::1] NHmax,
):
    """
    Perform ray tracing using sphere intersections.

    Parameters
    ----------
    xx: array
        sphere x coordinates
    yy: array
        sphere y coordinates
    zz: array
        sphere z coordinates
    RR: array
        sphere radii
    rho: array
        sphere densities (in units of 1e22/cm^2). For conversion from length to column density.
    x: array
        ray start coordinate, x component
    y: array
        ray start coordinate, y component
    z: array
        ray start coordinate, z component
    a: array
        ray direction, x component
    b: array
        ray direction, y component
    c: array
        ray direction, z component
    NHmax: array
        column density where the ray should terminate.

    Returns
    ----------
    t: array
        end position in multiples of the direction vector. -1 if infinite.
        Of shape len(x).
    """
    cdef int lenxx = len(xx)
    cdef int lena = len(a)
    cdef cnp.ndarray[double, ndim=1] t = np.zeros(lena)
    assert len(b) == lena
    assert len(c) == lena
    assert len(x) == lena
    assert len(y) == lena
    assert len(z) == lena
    assert len(yy) == lenxx
    assert len(zz) == lenxx
    assert len(RR) == lenxx
    assert len(rho) == lenxx
    assert len(NHmax) == lena
    cdef int r = sphere_raytrace_finite(
        &xx[0], &yy[0], &zz[0], &RR[0], &rho[0], lenxx, &x[0], &y[0], &z[0], &a[0], &b[0], &c[0], lena, &NHmax[0], &t[0])
    if r != 0:
        raise RuntimeError("sphere_raytrace_finite calculation failed")
    return t


@cython.boundscheck(False)
@cython.wraparound(False)
def py_cone_raytrace_finite(
    double[::1] thetas,
    double[::1] rhos,
    double[::1] x,
    double[::1] y,
    double[::1] z,
    double[::1] a,
    double[::1] b,
    double[::1] c,
    double[::1] d,
):
    """
    Perform ray tracing through a sphere/cone cuts.

    Sphere radius is 1, each cone angle defines a region of a certain density.

    This function raytraces from the starting coordinates in the direction
    given and compute the column density of the intersecting segments.
    Then it will go along the ray in the positive direction until the
    column density d is reached. The coordinates of that point are stored into
    (x, y, z).

    Parameters
    ----------
     regarding the cones:
    thetas: array
        the opening angles of the cones
    rhos: array
        density of the cones. For conversion from length to column density.
    x: array
        ray start coordinate, x component
    y: array
        ray start coordinate, y component
    z: array
        ray start coordinate, z component
    a: array
        ray direction, x component
    b: array
        ray direction, y component
    c: array
        ray direction, z component
    d: array
        column density where the ray should terminate.

    Returns
    ----------
    t: array
        end position in multiples of the direction vector. -1 if infinite.
        Of shape len(x).
    """
    cdef int n = thetas.shape[0]
    assert rhos.shape[0] == n
    cdef int lena = a.shape[0]
    assert len(b) == lena
    assert len(c) == lena
    assert len(x) == lena
    assert len(y) == lena
    assert len(z) == lena
    assert len(d) == lena
    cdef cnp.ndarray[double, ndim=1] t = np.zeros(lena)
    cdef int r = cone_raytrace_finite(
        &thetas[0], &rhos[0], n, &x[0], &y[0], &z[0], &a[0], &b[0], &c[0], lena, &d[0], &t[0])
    if r != 0:
        raise RuntimeError("cone_raytrace_finite calculation failed")
    return t


@cython.boundscheck(False)
@cython.wraparound(False)
def py_grid_raytrace_finite(
    cnp.ndarray[double, ndim=3] rho,
    double[::1] x,
    double[::1] y,
    double[::1] z,
    double[::1] a,
    double[::1] b,
    double[::1] c,
    double[::1] NHmax,
):
    """
    Ray tracing on a grid.

    Parameters
    ----------
    rho: array
        3D cube of densities (in units of 1e22/cm^2). For conversion from length to column density.
        Must be same size in each axis.
    x: array
        ray start coordinate, x component
    y: array
        ray start coordinate, y component
    z: array
        ray start coordinate, z component
    a: array
        ray direction, x component
    b: array
        ray direction, y component
    c: array
        ray direction, z component
    NHmax: array
        column density where the ray should terminate.

    Returns
    ----------
    t: array
        end position in multiples of the direction vector. -1 if infinite.
        Of shape len(x).
    """
    cdef int lena = len(a)
    cdef int lenrho = rho.shape[0]
    assert lenrho == rho.shape[1]
    assert lenrho == rho.shape[2]
    return py_grid_raytrace_finite_flat(np.array(rho.flatten()), lenrho, x, y, z, a, b, c, NHmax)


@cython.boundscheck(False)
@cython.wraparound(False)
def py_grid_raytrace_finite_flat(
    double[::1] rho_flat,
    int lenrho,
    double[::1] x,
    double[::1] y,
    double[::1] z,
    double[::1] a,
    double[::1] b,
    double[::1] c,
    double[::1] NHmax,
):
    """
    Ray tracing on a grid.

    Parameters
    ----------
    rho_flat: array
        flattened 3D cube of densities. For conversion from length to column density.
    lenrho: int
        length of each 3D cube axis
    x: array
        ray start coordinate, x component
    y: array
        ray start coordinate, y component
    z: array
        ray start coordinate, z component
    a: array
        ray direction, x component
    b: array
        ray direction, y component
    c: array
        ray direction, z component
    NHmax: array
        column density where the ray should terminate.

    Returns
    ----------
    t: array
        end position in multiples of the direction vector. -1 if infinite.
        Of shape len(x).
    """
    cdef int lena = len(a)
    assert len(b) == lena
    assert len(c) == lena
    assert len(x) == lena
    assert len(y) == lena
    assert len(z) == lena
    cdef cnp.ndarray[double, ndim=1] t = np.zeros(lena)
    cdef int r = grid_raytrace_finite(
        &rho_flat[0], lenrho, &x[0], &y[0], &z[0], &a[0], &b[0], &c[0], lena, &NHmax[0], &t[0])
    if r != 0:
        raise RuntimeError("grid_raytrace_finite calculation failed")
    return t
