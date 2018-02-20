from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension

extension = [Extension("non_max_s", ["non_max_s.pyx", "nms.cc"], language="c++",
             extra_compile_args=["-std=c++11"],
             extra_link_args=["-std=c++11"])]

setup(ext_modules = cythonize(extension))
