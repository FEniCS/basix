from skbuild import setup

setup(name="fenics-basix",
      python_requires='>=3.7.0',
      version="0.4.0",
      description='Basix Python interface',
      url="https://github.com/FEniCS/basix",
      author='FEniCS Project',
      author_email="fenics-dev@googlegroups.com",
      maintainer_email="fenics-dev@googlegroups.com",
      license="MIT",
      packages=["basix"],
      package_dir={"": "python"},
      cmake_install_dir="python/basix/")
