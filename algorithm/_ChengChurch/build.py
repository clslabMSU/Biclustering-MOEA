from Cython.Build import cythonize
from distutils.core import setup
#from os import name
from os.path import join

import numpy as np


if __name__ == "__main__":
    setup(
        name="mean_squared_residue",
        ext_modules=cythonize(join("algorithm", "_ChengChurch", "mean_squared_residue.pyx")),
        include_dirs=[np.get_include()]
    )
    """
    if name == "nt":
        setup(
            name="mean_squared_residue",
            ext_modules=cythonize("algorithm\\_ChengChurch\\mean_squared_residue.pyx")
        )
    else:
        setup(
            name="mean_squared_residue",
            ext_modules=cythonize("algorithm/_ChengChurch/mean_squared_residue.pyx")
        )"""