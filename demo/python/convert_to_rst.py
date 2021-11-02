import os
import pylit

pylit.defaults.text_extensions = [".rst"]

path = os.path.dirname(os.path.realpath(__file__))
for file in os.listdir(path):
    if file.endswith(".py") and file.startswith("demo"):
        pylit.main([file])
