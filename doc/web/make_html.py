import argparse
import markdown
import os
import basix
import re


_path = os.path.dirname(os.path.realpath(__file__))


def path(dir):
    return os.path.join(_path, dir)


temp = path("_temp")

parser = argparse.ArgumentParser(description="Build Basix documentation")
parser.add_argument('--url', metavar='url',
                    default="http://localhost", help="URL of built documentation")

args = parser.parse_args()
url = args.url
url_no_http = url.split("//")[1]


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
    if "{{SUPPORTED ELEMENTS}}" in txt:
        with open(path("../../README.md")) as f:
            info = "## Supported elements" + f.read().split("## Supported elements")[1].split("\n## ")[0]
        txt = txt.replace("{{SUPPORTED ELEMENTS}}", info)
    if "{{INSTALL}}" in txt:
        with open(path("../../INSTALL.md")) as f:
            info = f.read()
        txt = txt.replace("{{INSTALL}}", info)
    if "{{NAVBAR}}" in content:
        with open(os.path.join(temp, "navbar.html")) as f:
            info = f.read()
        txt = txt.replace("{{NAVBAR}}", info)
    if "{{STYLESHEETS}}" in content:
        with open(os.path.join(temp, "stylesheets.html")) as f:
            info = f.read()
        txt = txt.replace("{{STYLESHEETS}}", info)
    txt = txt.replace("{{URL}}", url)
    txt = txt.replace("{{VERSION}}", basix.__version__)
    txt = re.sub(r"\{\{link:URL/([^\}]*)\}\}", link_markup, txt)
    return txt


# Make paths
for dir in ["html", "cpp", "python"]:
    if os.path.isdir(path(dir)):
        os.system(f"rm -r {path(dir)}")
if os.path.isdir(temp):
    os.system(f"rm -r {temp}")
os.system(f"mkdir {path('html')}")
os.system(f"mkdir {temp}")

# Copy cpp and python folders
os.system(f"cp -r {path('../cpp')} {path('cpp')}")
os.system(f"cp -r {path('../python')} {path('python')}")
os.system(f"mkdir {path('python/source/_templates')}")

# Prepare templates
for file in ["stylesheets.html", "navbar.html", "intro.html", "outro.html"]:
    with open(os.path.join(path('template'), file)) as f:
        content = f.read()
    with open(os.path.join(temp, file), "w") as f:
        f.write(insert_info(content))

for fin, fout in [
    ("cpp-header.html.template", "cpp/header.html"),
    ("python-layout.html.template", "python/source/_templates/layout.html")
]:
    with open(path(fin)) as f:
        content = f.read()
    with open(path(fout), "w") as f:
        f.write(insert_info(content))

# Copy images and assets
os.system(f"cp -r {path('../../img')} {path('html')}/img")
os.system(f"cp -r {path('assets')} {path('html')}/assets")

# Convert markdown to html
for file in os.listdir(_path):
    if file.endswith(".md"):
        with open(path(file)) as f:
            contents = unicode_to_html(insert_info(f.read()))
        with open(os.path.join(path('html'), file[:-3] + ".html"), "w") as f:
            f.write(make_html_page(markdown.markdown(contents)))

# Make cpp docs
assert os.system(f"cd {path('cpp')} && doxygen && cp -r {path('cpp/html')} {path('html/cpp')}") == 0

# Make python docs
assert os.system(f"cd {path('python')} && make html && cp -r {path('python/build/html')} {path('html/python')}") == 0
