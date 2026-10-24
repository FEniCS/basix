# Copyright (c) 2022 Chris Richardson, Garth Wells and Jack S. Hale
# FEniCS Project
# SPDX-License-Identifier: MIT

import re

import pytest

import basix
import basix._basixcpp


def is_canonical(version):
    """Check version number is canonical according to PEP0440

    From https://peps.python.org/pep-0440/#appendix-b-parsing-version-strings-with-regular-expressions"""
    return (
        re.match(
            r"^([1-9][0-9]*!)?(0|[1-9][0-9]*)(\.(0|[1-9][0-9]*))*((a|b|rc)(0|[1-9][0-9]*))?(\.post(0|[1-9][0-9]*))?(\.dev(0|[1-9][0-9]*))?$",
            version,
        )
        is not None
    )


def test_version(python_version=basix.__version__, cpp_version=basix._basixcpp.__version__):
    assert is_canonical(python_version)

    # C++ marks a development version with a fourth component, and the
    # Python dev number is free so builds can be uniquely numbered.
    release = re.sub(r"\.(post|dev)(0|[1-9][0-9]*)", "", python_version)
    is_dev = ".dev" in python_version
    cpp_parts = cpp_version.split(".")
    if release != ".".join(cpp_parts[:3]) or is_dev != (len(cpp_parts) == 4):
        raise RuntimeError(
            f"The version numbers of the Python ({python_version}) and "
            f"nanobind/C++ ({cpp_version}) libraries do not match."
        )


def test_test_version_logic():
    with pytest.raises(RuntimeError):
        test_version("0.4.2.dev0", "0.4.2")

    with pytest.raises(RuntimeError):
        test_version("0.4.2", "0.4.2.0")
    with pytest.raises(RuntimeError):
        test_version("0.4.3.dev0", "0.4.2.0")

    test_version("0.4.2.post0", "0.4.2")
    test_version("0.4.2.dev0", "0.4.2.0")
    test_version("0.4.2.dev202610171644", "0.4.2.0")
