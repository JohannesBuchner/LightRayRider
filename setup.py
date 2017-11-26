import os.path

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

long_description = ""
with open('README.rst') as f:
    long_description = f.read()

setup(
    name='LightRayRider',
    version='1.0',
    author='Johannes Buchner',
    author_email='buchner.johannes@gmx.at',
    py_modules=['raytrace'],
    data_files=[('','ray.so'), ('','ray-parallel.so')],
    url='https://github.com/JohannesBuchner/imagehash',
    license='BSD 2-clause (see LICENSE file)',
    description='Ray Tracing Monte Carlo photon cropagator and column density calculator',
    long_description=long_description,
    install_requires=[
        "numpy",
        "scipy",       # for phash
    ],
    tests_require=['nose'],
    test_suite='nose.collector',
)
