import os

def test_cpp_vs_readme():
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../README.md") as f:
        readme = f.read().split("## Supported elements")[1].split("\n## ")[0]
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../cpp/doc/index.md") as f:
        cpp_docs = f.read().split("## Supported elements")[1].split("\n## ")[0]

    assert readme == cpp_docs
