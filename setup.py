import numpy
from setuptools import Extension
from Cython.Build import cythonize
#from distutils.extension import Extension
#from Cython.Distutils import build_ext
#from setuptools.command.build_py import build_py as _build_py

from setuptools import setup

with open('README.rst', encoding="utf-8") as readme_file:
    readme = readme_file.read()

compile_args = ['-std=c99', '-lm', '-Wall', '-Wextra', '-Wno-unused-function', '-DNPY_NO_DEPRECATED_API']

setup(
    author='Johannes Buchner',
    packages=['lightrayrider'],
    ext_modules = cythonize([
        Extension(
            "lightrayrider.parallel",
            ["lightrayrider/parallel.pyx"],
            include_dirs=[numpy.get_include()],
            extra_compile_args=compile_args + ["-fopenmp", "-DPARALLEL=1"],
            extra_link_args=compile_args + ["-fopenmp", "-DPARALLEL=1"],
        ),
        Extension(
            "lightrayrider.raytrace",
            ["lightrayrider/raytrace.pyx"],
            include_dirs=[numpy.get_include()],
            extra_compile_args=compile_args,
            extra_link_args=compile_args,
        )
    ]),
    long_description=readme,
)
