import os
import sys


def test_generated_docs():
    path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(path, "../python/docs.h")) as f:
        docs = f.read()

    sys.path.append(os.path.join(path, "../python"))
    from generate_docs import generate_docs
    assert generate_docs() == docs
