# Installation instructions

## Using pip

The preferred method of installation is to use pip. When using pip, the build dependencies should
be downloaded automatically, with the exception of Eigen3, which needs to be installed beforehand.

Eigen3 is available at: https://gitlab.com/libeigen/eigen/-/releases or commonly via package managers,
e.g. in Ubuntu as libeigen3-dev.

pip3 install .


## Using setup.py

The legacy setup.py install requires several packages to be preinstalled:

scikit-build, cmake>=3.18, ninja, eigen>=3.3.7, pybind11>=2.6.0

Once these are on your system, it should be possible to install with:

python3 setup.py install --prefix=<install_prefix>


## Using cmake

Several prerequisites include: cmake, eigen3 and pybind11
(FIXME: check exactly what is required)

e.g.

cmake -DCMAKE_INSTALL_PREFIX=<install_prefix> .
make -j 3 install
