from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

sourcefiles = ['env_ex.pyx']

setup(
  cmdclass = {'build_ext' : build_ext},
  ext_modules = [Extension("env_ex", sourcefiles)]
)
