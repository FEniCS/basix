import argparse
from markdown import markdown as _markdown
import os
import basix
import re
import yaml

_path = os.path.dirname(os.path.realpath(__file__))


def path(dir):
    return os.path.join(_path, dir)


temp = path("_temp")

parser = argparse.ArgumentParser(description="Build Basix documentation")
parser.add_argument('--url', metavar='url',
                    default="http://localhost", help="URL of built documentation")
parser.add_argument('--clone', metavar='clone',
                    default="true", help="Clone the web repository")

args = parser.parse_args()
url = args.url
url_no_http = url.split("//")[1]


def markdown(txt):
    """Convert markdown to HTML."""
    # Convert code inside ```s (as these are not handled by the markdown library
    tsp = txt.split("```")
    out = ""
    code = False
    for part in tsp:
        if code:
            part = part.strip().split("\n")
            if part[0] in ["console", "bash", "python", "c", "cpp"]:
                part = part[1:]
            part = "\n".join(part)
            out += f"<pre><code>{part}</code></pre>"
        else:
            out += part
        code = not code
    return _markdown(out)


def jekyll(a):
    """
    Convert Jekyll to HTML.

    This is used to convert template pages from the FEniCS website to the format needed
    here.
    """
    while "{% include" in a:
        b, c = a.split("{% include", 1)
        c, d = c.split("%}", 1)
        with open(path(f"web/_includes/{c.strip()}")) as f:
            a = b + f.read() + d

    while "{% for link in site.data.navbar %}" in a:
        b, c = a.split("{% for link in site.data.navbar %}", 1)
        c, d = c.split("{% endfor %}", 1)
        with open(path("web/_data/navbar.yml")) as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        a = b
        for i in data:
            this_c = c
            for j, k in i.items():
                if j == "page" and k.startswith("/"):
                    print(j, k)
                    this_c = this_c.replace(f"{{{{ link.{j} }}}}", f"https://fenicsproject.org{k}")
                else:
                    this_c = this_c.replace(f"{{{{ link.{j} }}}}", k)
            a += this_c
        a += d
    if "{% if page.image %}" in a:
        b, c = a.split("{% if page.image %}")
        _, c = c.split("{% else %}", 1)
        c, d = c.split("{% endif %}", 1)
        a = b + c + d

    if "{{ page.title | default: site.title }}" in a:
        with open(path("template/title.html")) as f:
            a = a.replace("{{ page.title | default: site.title }}", f.read())
    if "\n{% if page.subtitle %}" in a:
        b, c = a.split("\n{% if page.subtitle %}")
        _, c = c.split("\n{% endif %}", 1)
        a = b + c

    a = re.sub(r"\{\{[^}]+\| default: (?:'|\")([^}]+)(?:'|\") \}\}", r"\1", a)
    a = re.sub(r"\{\{ (?:'|\")\/assets\/css\/style\.css\?v=(?:'|\")[^\}]+\}\}", "/assets/css/style.css", a)
    a = a.replace("{% seo %}", "{{SEO}}")
    a = re.sub(r"([\"'\(])/assets", r"\1https://fenicsproject.org/assets", a)

    assert "{%" not in a

    return a


def unicode_to_html(txt):
    txt = txt.replace(u"\xe9", "&eacute;")
    return txt


def make_html_page(html):
    with open(os.path.join(temp, "intro.html")) as f:
        pre = f.read()
    with open(os.path.join(temp, "outro.html")) as f:
        post = f.read()
    return pre + html + post


def link_markup(matches):
    return f"[{url_no_http}/{matches[1]}]({url}/{matches[1]})"


def insert_info(txt):
    if "{{SUMMARY}}" in txt:
        started = False
        info = ""
        with open(path("../../README.md")) as f:
            for line in f:
                if not started and line.strip() != "" and line[0] not in ["!", "#"]:
                    started = True
                if started:
                    if line.startswith("#"):
                        break
                    info += line
        info = info.replace("joss/img", "img")
        txt = txt.replace("{{SUMMARY}}", info)
    if "{{SUPPORTED ELEMENTS}}" in txt:
        with open(path("../../README.md")) as f:
            info = "## Supported elements" + f.read().split("## Supported elements")[1].split("\n## ")[0]
        info = info.replace("joss/img", "img")
        txt = txt.replace("{{SUPPORTED ELEMENTS}}", info)
    if "{{INSTALL}}" in txt:
        with open(path("../../INSTALL.md")) as f:
            info = f.read()
        txt = txt.replace("{{INSTALL}}", info)
    txt = txt.replace("{{URL}}", url)
    txt = txt.replace("{{VERSION}}", basix.__version__)
    txt = re.sub(r"\{\{link:URL/([^\}]*)\}\}", link_markup, txt)
    return txt


def system(command):
    assert os.system(command) == 0


# Make paths
for dir in ["html", "cpp", "python"]:
    if os.path.isdir(path(dir)):
        system(f"rm -r {path(dir)}")
if os.path.isdir(temp):
    system(f"rm -r {temp}")
system(f"mkdir {path('html')}")
system(f"mkdir {temp}")

# Copy cpp and python folders
system(f"cp -r {path('../cpp')} {path('cpp')}")
system(f"cp -r {path('../python')} {path('python')}")
system(f"mkdir {path('python/source/_templates')}")

# Prepare templates
if args.clone == "true":
    if os.path.isdir(path('web')):
        system(f"rm -r {path('web')}")
    system(f"git clone http://github.com/FEniCS/web {path('web')}")

with open(path('web/_layouts/default.html')) as f:
    intro, outro = f.read().split("\n    {{ title }}\n    {{ content }}\n")
    intro = insert_info(jekyll(intro))
    outro = insert_info(jekyll(outro))

with open(path("template/navbar.html")) as f:
    intro += f"<h2 id=\"project_subtitle\">{insert_info(f.read())}</h2>"

with open(os.path.join(temp, "intro.html"), "w") as f:
    f.write(intro.replace("{{SEO}}", "<title>Basix documentation</title>"))
with open(os.path.join(temp, "outro.html"), "w") as f:
    f.write(outro)

intro += "\n".join(outro.split("\n")[:2])
outro = "\n".join(outro.split("\n")[2:])

with open(path("cpp/footer.html"), "w") as f:
    f.write(outro)
with open(path("cpp/header.html"), "w") as f:
    with open(path("cpp-seo.html")) as f2:
        f.write(insert_info(intro.replace("{{SEO}}", f2.read())))
    with open(path("cpp-header.html")) as f2:
        f.write(insert_info(f2.read()))

content = "{% extends \"!layout.html\" %}\n"
content += "{%- block content %}\n"
content += intro.split("<body>")[1] + "\n"
content += "    {{ super() }}\n"
content += outro.split("</body>")[0] + "\n"
content += "{% endblock %}\n"
content += "{%- block extrahead %}\n"
content += intro.split("<head>")[1].split("</head>")[0] + "\n"
content += "  {{ super() }}\n"
content += "{% endblock %}"

with open(path("python/source/_templates/layout.html"), "w") as f:
    f.write(insert_info(content))

with open(path("cpp/Doxyfile")) as f:
    content = ""
    for line in f:
        if line.startswith("HTML_HEADER"):
            content += f"{line.split('=')[0]}= header.html\n"
        elif line.startswith("HTML_FOOTER"):
            content += f"{line.split('=')[0]}= footer.html\n"
        else:
            content += line.replace(" ..", " ../..")

with open(path("cpp/Doxyfile"), "w") as f:
    f.write(content)

# Copy images and assets
system(f"cp -r {path('../../joss/img')} {path('html')}/img")
system(f"cp -r {path('assets')} {path('html')}/assets")

# Convert markdown to html
for file in os.listdir(_path):
    if file.endswith(".md"):
        with open(path(file)) as f:
            contents = unicode_to_html(insert_info(f.read()))
        with open(os.path.join(path('html'), file[:-3] + ".html"), "w") as f:
            f.write(make_html_page(markdown(contents)))

# Make cpp docs
system(f"cd {path('cpp')} && doxygen")
system(f"cp -r {path('cpp/html')} {path('html/cpp')}")

# Make demos
os.system(f"rm {path('../../demo/python/*.py.rst')}")
os.system(f"rm {path('../../demo/python/*.png')}")
# If file saves matplotlib images, run the demo
for file in os.listdir(path("../../demo/python")):
    if file.endswith(".py"):
        with open(path(f"../../demo/python/{file}")) as f:
            content = f.read()
        if "savefig" in content:
            here = os.getcwd()
            os.chdir(path("../../demo/python"))
            system(f"python3 {file}")
            os.chdir(here)
system(f"cd {path('../../demo/python')} && python3 convert_to_rst.py")
system(f"mkdir {path('python/source/demo')}")
system(f"cp {path('../../demo/python/*.rst')} {path('python/source/demo')}")
system(f"cp {path('../../demo/python/*.png')} {path('python/source/demo')}")
for file in os.listdir(path("python/source/demo")):
    if file.endswith(".py.rst"):
        with open(path(f"python/source/demo/{file}")) as f:
            content = f.read()
        content = content.replace("literalinclude:: ../", "literalinclude:: ../../../../../demo/")
        with open(path(f"python/source/demo/{file}"), "w") as f:
            f.write(content)

with open(path("python/source/index.rst"), "a") as f:
    f.write("\n")
    f.write(".. toctree::\n")
    f.write("   demo/index")

# Make python docs
system(f"cd {path('python')} && make html")
system(f"cp -r {path('python/build/html')} {path('html/python')}")
