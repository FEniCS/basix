# Copyright (c) 2022 Chris Richardson, Garth Wells and Jack S. Hale
# FEniCS Project
# SPDX-License-Identifier: MIT

import pkg_resources
import re
import basix


def is_canonical(version):
    """Check version number is canonical according to PEP0440

    From https://peps.python.org/pep-0440/#appendix-b-parsing-version-strings-with-regular-expressions"""
    return re.match(r'^([1-9][0-9]*!)?(0|[1-9][0-9]*)(\.(0|[1-9][0-9]*))*((a|b|rc)(0|[1-9][0-9]*))?(\.post(0|[1-9][0-9]*))?(\.dev(0|[1-9][0-9]*))?$', version) is not None  # noqa: E501


def test_version():
    version = pkg_resources.get_distribution("fenics-basix").version

    assert is_canonical(version)

    # Strip Python-specific versioning (dev, post) and compare with C++
    # versioning
    stripped_version = re.sub(r"(\.post(0|[1-9][0-9]*))", "", version)
    stripped_version = stripped_version.replace("dev", "")
    if stripped_version != basix.__version__:
        raise RuntimeError(
            f"The version numbers of the Python ({pkg_resources.get_distribution('fenics-basix').version} "
            + f"-> {stripped_version}) and pybind11/C++ ({basix.__version__}) libraries does not match")
