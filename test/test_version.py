# Copyright (c) 2021 Chris Richardson and Garth Wells
# FEniCS Project
# SPDX-License-Identifier: MIT

import pkg_resources
import re
import basix

# From https://peps.python.org/pep-0440/#appendix-b-parsing-version-strings-with-regular-expressions


def is_canonical(version):
    return re.match(r'^([1-9][0-9]*!)?(0|[1-9][0-9]*)(\.(0|[1-9][0-9]*))*((a|b|rc)(0|[1-9][0-9]*))?(\.post(0|[1-9][0-9]*))?(\.dev(0|[1-9][0-9]*))?$', version) is not None  # noqa: E501


def test_version():
    version = pkg_resources.get_distribution("fenics-basix").version

    assert is_canonical(version)

    # Remove Python-specific versioning (dev, post) and compare with C++
    # versioning
    version = version.replace('dev', '')
    version = re.sub(r"(\.post(0|[1-9][0-9]*))", "", version)
    version = re.sub(r"(\.dev(0|[1-9][0-9]*))", "", version)
    if version != basix.__version__:
        raise RuntimeError("Incorrect installation version compared to pybind")
