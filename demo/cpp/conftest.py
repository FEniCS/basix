import sys


def pytest_addoption(parser):
    parser.addoption(
        "--cmake-args",
        action="store",
        default=f"-DPython3_EXECUTABLE={sys.executable}",
        help="arguments to pass to cmake configure",
    )
