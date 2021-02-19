import os

for file in os.listdir("."):
    if file[0] != "." and file[-3:] != ".py":
        new_file = ""
        with open(file) as f:
            for line in f:
                if line.startswith("#include \"basix/core/"):
                    new_line = "#include \"" + line[16:-2] + "\""
                    new_file += new_line + "\n"
                    print(new_line)
                else:
                    new_file += line
        with open(file, "w") as f:
            f.write(new_file)
