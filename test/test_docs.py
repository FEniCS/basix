# Copyright (c) 2021 Matthew Scroggs
# FEniCS Project
# SPDX-License-Identifier: MIT

import os
import pytest
import sys


def test_generated_docs():
    # If this test fails, you should run `python generate_docs.py` in the
    # python/ folder to re-generate the docs for the pybind interface

    path = os.path.dirname(os.path.realpath(__file__))
    if not os.path.isfile(os.path.join(path, "../python/docs.h")):
        pytest.skip("This test can only be run from the source directory.")

    with open(os.path.join(path, "../python/docs.h")) as f:
        docs = f.read()

    sys.path.append(os.path.join(path, "../python"))
    from generate_docs import generate_docs
    assert generate_docs() == docs


def test_demo_index():
    path = os.path.dirname(os.path.realpath(__file__))
    if not os.path.isfile(os.path.join(path, "../demo/python/index.rst")):
        pytest.skip("This test can only be run from the source directory.")

    with open(os.path.join(path, "../demo/python/index.rst")) as f:
        index = f.read()

    for file in os.listdir(os.path.join(path, "../demo/python")):
        if file.endswith(".py") and file.startswith("demo_"):
            assert file in index
