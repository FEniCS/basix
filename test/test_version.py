# Copyright (c) 2021-2022 Chris Richardson, Garth Wells, Matthew Scroggs
# FEniCS Project
# SPDX-License-Identifier: MIT

import pytest
import pkg_resources
import basix
import os
import re


def test_version():
    version = pkg_resources.get_distribution("fenics-basix").version
    version = version.replace('dev', '')
    if version != basix.__version__:
        raise RuntimeError("Incorrect installation version compared to pybind")


def test_version_numbering():
    py_version = pkg_resources.get_distribution("fenics-basix").version
    cpp_version = py_version.replace('dev', '')

    path = os.path.dirname(os.path.realpath(__file__))
    if not os.path.isfile(os.path.join(path, "../python/docs.h")):
        pytest.skip("This test can only be run from the source directory.")


    for file in ["../CMakeLists.txt", "../cpp/CMakeLists.txt", "../python/CMakeLists.txt"]:
        print(f"Checking version numbers in {file}.")

        with open(os.path.join(path, file)) as f:
            content = f.read()
        for m in re.findall(r"(?:(?:VERSION)|(?:version))[\s=]+([\"'])(.+?)\1", content):
            assert m[1] == cpp_version

    for file in ["../setup.py", "../python/setup.py"]:
        print(f"Checking version numbers in {file}.")

        with open(os.path.join(path, file)) as f:
            content = f.read()
        for m in re.findall(r"(?:(?:VERSION)|(?:version))[\s=]+([\"'])(.+?)\1", content):
            assert m[1] == py_version
