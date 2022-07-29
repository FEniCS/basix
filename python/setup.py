import os
import subprocess
import sys
import sysconfig

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

if sys.version_info < (3, 7):
    print("Python 3.7 or higher required, please upgrade.")
    sys.exit(1)

CLASSIFIERS = """\
Development Status :: 5 - Production/Stable
Intended Audience :: Developers
Intended Audience :: Science/Research
License :: OSI Approved :: MIT License
Operating System :: POSIX
Operating System :: POSIX :: Linux
Operating System :: MacOS :: MacOS X
Programming Language :: Python
Programming Language :: Python :: 3
Programming Language :: Python :: 3.7
Programming Language :: Python :: 3.8
Programming Language :: Python :: 3.9
Programming Language :: Python :: 3.10
Topic :: Scientific/Engineering :: Mathematics
Topic :: Software Development :: Libraries :: Python Modules
"""


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: "
                               + ", ".join(e.name for e in self.extensions))

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPython3_EXECUTABLE=' + sys.executable,
                      '-DPython3_LIBRARIES=' + sysconfig.get_config_var("LIBDEST"),
                      '-DPython3_INCLUDE_DIRS=' + sysconfig.get_config_var("INCLUDEPY")]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
        build_args += ['--', '-j3']

        env = os.environ.copy()
        import pybind11
        env['pybind11_DIR'] = pybind11.get_cmake_dir()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                              self.distribution.get_version())

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp, env=env)


setup(name='fenics-basix',
      version='0.5.0',
      description='Basix Python interface',
      url="https://github.com/FEniCS/basix",
      author='FEniCS Project',
      author_email="fenics-dev@googlegroups.com",
      maintainer_email="fenics-dev@googlegroups.com",
      license="MIT License",
      classifiers=[_f for _f in CLASSIFIERS.split('\n') if _f],
      platforms=["Linux", "Mac OS-X", "Unix"],
      packages=["basix"],
      install_requires=["numpy>=1.21"],
      package_data={"basix": ["py.typed"]},
      setup_requires=["pybind11"],
      ext_modules=[CMakeExtension('basix._basixcpp')],
      cmdclass=dict(build_ext=CMakeBuild),
      zip_safe=False)
