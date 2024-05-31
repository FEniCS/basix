# Re-generating nanobind stubs

To update the nanobind type stubs, after installing the Basix Python
interface, run:
```sh
python -m nanobind.stubgen -m basix._basixcpp -M basix/py.typed -p pattern_stub.txt -o basix/_basixcpp.pyi
```
from the `python/` directory.