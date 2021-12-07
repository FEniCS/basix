from skbuild import setup

import sys
import sysconfig

setup(name="fenics-basix",
      python_requires='>=3.7.0',
      version="0.3.1.dev0",
      description='Basix Python interface',
      url="https://github.com/FEniCS/basix",
      author='FEniCS Project',
      author_email="fenics-dev@googlegroups.com",
      maintainer_email="fenics-dev@googlegroups.com",
      license="MIT",
      packages=["basix"],
      cmake_args=[
          '-DPython3_EXECUTABLE=' + sys.executable,
          '-DPython3_LIBRARIES=' + sysconfig.get_config_var("LIBDEST"),
          '-DPython3_INCLUDE_DIRS=' + sysconfig.get_config_var("INCLUDEPY"),
          '-DXLIBS_DOWNLOAD=ON'],
      package_dir={"": "python"},
      cmake_install_dir="python/basix/")
