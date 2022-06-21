# Copyright (c) 2021-2022 Chris Richardson, Garth Wells, Matthew Scroggs, Jack S. Hale
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

    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
    if not os.path.isfile(os.path.join(path, "LICENSE")):
        pytest.skip("This test can only be run from the source directory.")

    for file in ["CMakeLists.txt", "cpp/CMakeLists.txt", "python/CMakeLists.txt"]:
        print(f"Checking version numbers in {file}.")

        with open(os.path.join(path, file)) as f:
            content = f.read()
        matches = re.findall(r"(?:(?:VERSION)|(?:version))[\s=]+([\"'])(.+?)\1", content)
        assert len(matches) > 0
        for m in matches:
            assert m[1] == cpp_version

    for file in ["setup.py", "python/setup.py"]:
        print(f"Checking version numbers in {file}.")

        with open(os.path.join(path, file)) as f:
            content = f.read()
        matches = re.findall(r"(?:(?:VERSION)|(?:version))[\s=]+([\"'])(.+?)\1", content)
        assert len(matches) > 0
        for m in matches:
            assert m[1] == py_version


def is_canonical(version):
    """Check version number is canonical according to PEP0440

    From https://peps.python.org/pep-0440/#appendix-b-parsing-version-strings-with-regular-expressions"""
    return re.match(r'^([1-9][0-9]*!)?(0|[1-9][0-9]*)(\.(0|[1-9][0-9]*))*((a|b|rc)(0|[1-9][0-9]*))?(\.post(0|[1-9][0-9]*))?(\.dev(0|[1-9][0-9]*))?$', version) is not None  # noqa: E501


def test_version(python_version=pkg_resources.get_distribution("fenics-basix").version, cpp_version=basix.__version__):
    assert is_canonical(python_version)

    # Strip Python-specific versioning (dev, post) and compare with C++
    # versioning
    stripped_version = re.sub(r"(\.post(0|[1-9][0-9]*))", "", python_version)
    stripped_version = stripped_version.replace("dev", "")
    if stripped_version != cpp_version:
        raise RuntimeError(
            f"The version numbers of the Python ({pkg_resources.get_distribution('fenics-basix').version} "
            + f"-> {stripped_version}) and pybind11/C++ ({basix.__version__}) libraries does not match")


def test_test_version_logic():
    with pytest.raises(RuntimeError):
        test_version("0.4.2.dev0", "0.4.2")

    test_version("0.4.2.post0", "0.4.2")
    test_version("0.4.2.dev0", "0.4.2.0")
