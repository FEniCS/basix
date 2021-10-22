import os
import pytest
import sys


def test_generated_docs():
    path = os.path.dirname(os.path.realpath(__file__))
    if not os.path.isfile(os.path.join(path, "../python/docs.h")):
        pytest.skip("This test can only be run from the source directory.")

    with open(os.path.join(path, "../python/docs.h")) as f:
        docs = f.read()

    sys.path.append(os.path.join(path, "../python"))
    from generate_docs import generate_docs
    assert generate_docs() == docs
