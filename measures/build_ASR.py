from distutils.core import setup
import numpy as np
from os import chdir, getcwd, name
from os.path import join
from Cython.Build import cythonize

MODULE = "ASR"

if name == "nt":
    print("Building on Windows... (dir=%s)" % getcwd()) # unfortunately
    filename = "\\".join(__file__.split("\\")[:-1]) + "\\%s.pyx" % MODULE
    chdir("\\".join(__file__.split("\\")[:-2]))
    setup(
        name=MODULE,
        ext_modules=cythonize([filename], annotate=True),
        include_dirs=[np.get_include()]
    )
else:
    setup(
        name=MODULE,
        ext_modules=cythonize([join("measures", "%s.pyx" % MODULE)], annotate=True),
        include_dirs=[np.get_include()]
    )