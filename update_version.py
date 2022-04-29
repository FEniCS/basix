"""Utility to update the version number in all locations.

Example usage
-------------
To update the version numbers in all files to "1.0.0", run either of the following:
```bash
python3 update_version.py -v 1.0.0
python3 update_version.py --version 1.0.0
```

To update the C++ version numbers to "1.0.0.3" and the Python version numbers to "1.0.0dev3", run
either of the following:
```bash
python3 update_version.py -v 1.0.0.dev3
python3 update_version.py --version 1.0.0.dev3
```
"""

import argparse
import os
import re


def replace_version(content, version):
    content = re.sub(r"((?:VERSION)|(?:version))([\s=]+)([\"']).+?\3",
                     lambda matches: f"{matches[1]}{matches[2]}{matches[3]}{version}{matches[3]}", content)
    content = re.sub(r"(\s+)(\"?)fenics-((?:basix)|(?:ffcx)|(?:dolfinx))\>\=.+(\2|\n)",
                     lambda matches: f"{matches[1]}{matches[2]}fenics-{matches[3]}>={version},<{next_version}{matches[4]}",
                     content)
    return content


parser = argparse.ArgumentParser(description="Update version numbering")
parser.add_argument('-v', '--version', metavar='version', help="Version number to update to", required=True)

args = parser.parse_args()

version = args.version
pyversion = version
if ".dev" in version:
    version = version.replace("dev", "")

print("About to update version numbers to:")
print(f"    C++ version: {version}")
print(f"    Python version: {pyversion}")

answer = None
while answer not in ["Y", "N"]:
    if answer is None:
        answer = input("Do you want to proceed? [Y/N] ").upper()
    else:
        answer = input("Please enter Y or N: ").upper()

if answer == "N":
    print("Aborting.")
    exit()


path = os.path.dirname(os.path.realpath(__file__))

for file in ["CMakeLists.txt", "cpp/CMakeLists.txt", "python/CMakeLists.txt"]:
    print(f"Replacing version numbers in {file}.")
    with open(os.path.join(path, file)) as f:
        content = f.read()
    with open(os.path.join(path, file), "w") as f:
        f.write(replace_version(content, version))

for file in ["setup.py", "python/setup.py"]:
    print(f"Replacing version numbers in {file}.")
    with open(os.path.join(path, file)) as f:
        content = f.read()
    with open(os.path.join(path, file), "w") as f:
        f.write(replace_version(content, pyversion))
