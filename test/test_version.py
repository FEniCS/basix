# Copyright (c) 2021 Chris Richardson and Garth Wells
# FEniCS Project
# SPDX-License-Identifier: MIT

import pkg_resources
import basix


def test_version():
    version = pkg_resources.get_distribution("fenics-basix").version
    version = version.replace('dev', '')
    if version != basix.__version__:
        raise RuntimeError("Incorrect installation version compared to pybind")
