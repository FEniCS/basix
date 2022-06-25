"""Generate docs for the pybind layer from the C++ header files."""

import os
import re

path = os.path.dirname(os.path.realpath(__file__))
cpp_path = os.path.join(path, "basix")

replacements = [(";", "@semicolon@"), ("{", "@openbrace@"), ("}", "@closebrace@"),
                ("<", "@opentri@"), (">", "@closetri@"), ("(", "@openb@"), (")", "@closeb@"),
                ("[", "@opensq@"), ("]", "@closesq@")]


def replace(txt):
    for i, j in replacements:
        txt = txt.replace(i, j)
    return txt


def unreplace(txt):
    for i, j in replacements:
        txt = txt.replace(j, i)
    return txt


def remove_types(matches):
    """Remove the types from a function declaration."""
    vars = [i.strip().split(" ")[-1].split("=")[0] for i in matches[1].split(",")]
    return "(" + ", ".join(vars) + ")"


def prepare_cpp(content):
    """Prepare a cpp file for parsing."""
    out = ""

    for line in content.split("\n"):
        if not line.strip().startswith("#"):
            if line.strip().startswith("//"):
                out += replace(line) + "\n"
            else:
                out += line + "\n"

    out = re.sub(r"namespace [^{]*\{", ";", out)
    out = re.sub(r"class [^{]*\{", "", out)

    while "{" in out:
        out = re.sub(r"\{[^{}]*\}", ";", out)
    while "<" in out and ">" in out:
        out = re.sub(r"<[^<>]*>", "", out)
    out = re.sub(r"\(([^\)]*)\)", remove_types, out)
    return out


def get_docstring(matches):
    """Get documentation from a header file."""
    # Parse the info
    info = matches[1]
    typename = None
    if " : " in info:
        info, typename = info.split(" : ")
    file, function, info = info.split(" > ", 2)
    if " > " in info:
        info_type, info = info.split(" > ")
    else:
        info_type = info

    # Read documentation from a header file
    with open(os.path.join(cpp_path, file)) as f:
        content = prepare_cpp(f.read())

    if "::" in function:
        function = function.split("::")[-1]
    if "(" not in function:
        function += "("

    if function not in content:
        print(function)
    assert function in content
    doc = content.split(function)[0].split(";")[-1]

    # Convert doxygen syntax to Python docs

    doc = "\n".join([i.strip()[4:] for i in doc.split("\n") if i.strip().startswith("///")])
    doc = doc.replace("@f$", "$")
    doc = doc.replace("@f[", "\\[")
    doc = doc.replace("@f]", "\\]")
    doc = doc.replace("@note", "NOTE:")
    doc = doc.replace("@todo", "TODO:")
    doc = doc.replace("@warning", "WARNING:")
    doc = unreplace(doc)

    if info_type == "doc":
        assert typename is None
        docstring = doc.split("@param")[0].split("@return")[0].strip()
        doclines = docstring.split("\n")
        in_code = False
        for i, j in enumerate(doclines):
            if re.match(r"^~+$", j):
                if in_code:
                    in_code = False
                    doclines[i] = "\n"
                else:
                    in_code = True
                    doclines[i] = "\n.. code-block::\n"
            elif in_code:
                doclines[i] = " " + doclines[i]
            else:
                doclines[i] = doclines[i].strip()
        for i, j in enumerate(doclines[:-1]):
            if re.match(r"^-+$", doclines[i + 1]):
                doclines[i] = f"*{j}*"
                doclines[i + 1] = ""
        return "\n".join(doclines)

    if info_type == "param":
        params = {}
        for i in doc.split("@param")[1:]:
            i = i.split("@return")[0]
            i = i.replace("[in]", "")
            i = i.replace("[out]", "")
            i = i.replace("[in,out]", "")
            i = " ".join([j.strip() for j in i.strip().split("\n")])
            if " " in i:
                p, pdoc = i.split(" ", 1)
            else:
                p = i
                pdoc = "TODO: document this"
            params[p] = "\n        ".join(pdoc.split("\n"))
        return f"{info}: {params[info]}"

    if info_type == "return":
        returns = [i.split("@param")[0].strip() for i in doc.split("@return")[1:]]
        if len(returns) == 0:
            returns.append("TODO: document this")
        assert len(returns) == 1
        returns = "\n    ".join(returns[0].split("\n"))
        return f"{returns}"


def generate_docs():
    with open(os.path.join(path, "docs.template")) as f:
        docs = f.read()

    docs = docs.replace("{{DOCTYPE}}", "const std::string")
    docs = re.sub(r"\{\{(.+?)\}\}", get_docstring, docs)

    return docs


if __name__ == "__main__":
    print(generate_docs())
