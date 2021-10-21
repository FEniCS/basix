import os
import re

path = os.path.dirname(os.path.realpath(__file__))
cpp_path = os.path.join(path, "../cpp/basix")

exclude = ["math.h"]


def cpp_to_python(doc):
    doc = doc.replace("@f$", "$")
    doc = doc.replace("@f[", "\\[")
    doc = doc.replace("@f]", "\\]")
    doc = doc.replace("@note", "NOTE:")
    doc = doc.replace("@todo", "TODO:")
    out = doc.split("@param")[0].split("@return")[0].strip()
    params = [i.split("@return")[0] for i in doc.split("@param")[1:]]
    returns = [i.split("@param")[0] for i in doc.split("@return")[1:]]
    out += "\n"
    if len(params) > 0:
        out += "\nParameters\n=========\n"
        for p in params:
            p = p.replace("[in]", "")
            p = p.replace("[out]", "")
            p = p.replace("[in,out]", "")
            try:
                name, desc = p.strip().split(" ", 1)
            except ValueError:
                name = p.strip()
                desc = "TODO: document this"
            out += name
            out += "\n"
            out += "\n".join(["    " + i.strip() for i in desc.strip().split("\n")])
            out += "\n"
    if len(returns) > 0:
        assert len(returns) == 1
        out += "\nReturns\n=======\n"
        out += "\n".join([i.strip() for i in returns[0].strip().split("\n")])
        out += "\n"
    return out


def generate_docs():
    docs = {}

    for file in os.listdir(cpp_path):
        if file.endswith(".h") and file not in exclude:
            with open(os.path.join(cpp_path, file)) as f:
                contents = ""
                for line in f:
                    line = line
                    if line.strip().startswith("//"):
                        contents += line.replace(";", "{{SEMICOLON}}")
                    else:
                        contents += line.replace("{", ";")

                for part in contents.split(";"):
                    if "///" in part:
                        doc = ""
                        new = False
                        for line in part.split("\n"):
                            if line.strip().startswith("///"):
                                if new:
                                    doc = ""
                                    new = False
                                doc += line.strip()[3:].strip() + "\n"
                            elif line.strip() != "":
                                new = True
                        doc = doc.replace("{{SEMICOLON}}", ";")

                        function_name = " ".join([i for i in part.split("\n") if not i.strip().startswith("//")])
                        function_name = function_name.replace("const", "")
                        function_name = re.sub(r"\([^\)]*\)", "", function_name)
                        function_name = function_name.strip().split(" ")[-1]

                        if "::" not in function_name:
                            doc_name = file[:-2].replace("-", "_") + "__" + function_name
                            assert re.match(r"^[A-Za-z_][A-Za-z0-9_]+$", doc_name)

                            if doc_name not in docs:
                                docs[doc_name] = []
                            docs[doc_name].append([doc, part])

    output = "#include <string>\n\n"
    output += "namespace basix::docstring\n{\n\n"
    for i, j in docs.items():
        if len(j) == 1:
            doc = cpp_to_python(j[0][0])
            if "\n" in doc:
                first, rest = doc.split("\n", 1)
            else:
                first = doc
                rest = ""
            for part in ["const std::string", f" {i}", f" = R\"({first}"]:
                if len(output.split("\n")[-1] + part) > 80:
                    output += "\n   "
                output += part
            output += "\n" + rest + ")\";\n\n"
        else:
            pass  # print("overloaded:", i, j)

    output += "} // namespace basix::docstring\n"
    return output


if __name__ == "__main__":
    with open(os.path.join(path, "docs.h"), "w") as f:
        f.write(generate_docs())
