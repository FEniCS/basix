import argparse
import markdown
import os
import basix
import re


path = os.path.dirname(os.path.realpath(__file__))
html_path = os.path.join(path, "html")
img_path = os.path.join(path, "../img")
assets_path = os.path.join(path, "assets")
template_path = os.path.join(path, "template")
temp_path = os.path.join(path, "_temp")

parser = argparse.ArgumentParser(description="Build defelement.com")
parser.add_argument('--url', metavar='url',
                    default="http://localhost", help="URL of built documentation")

args = parser.parse_args()
url = args.url
url_no_http = url.split("//")[1]


def unicode_to_html(txt):
    txt = txt.replace(u"\xe9", "&eacute;")
    return txt


def make_html_page(html):
    with open(os.path.join(temp_path, "intro.html")) as f:
        pre = f.read()
    with open(os.path.join(temp_path, "outro.html")) as f:
        post = f.read()
    return pre + html + post


def link_markup(matches):
    return f"[{url_no_http}/{matches[1]}]({url}/{matches[1]})"


def insert_info(txt):
    if "{{SUPPORTED ELEMENTS}}" in txt:
        with open(os.path.join(path, "../README.md")) as f:
            info = "## Supported elements" + f.read().split("## Supported elements")[1].split("\n## ")[0]
        txt = txt.replace("{{SUPPORTED ELEMENTS}}", info)
    if "{{INSTALL}}" in txt:
        with open(os.path.join(path, "../INSTALL.md")) as f:
            info = f.read()
        txt = txt.replace("{{INSTALL}}", info)
    if "{{NAVBAR}}" in content:
        with open(os.path.join(temp_path, "navbar.html")) as f:
            info = f.read()
        txt = txt.replace("{{NAVBAR}}", info)
    if "{{STYLESHEETS}}" in content:
        with open(os.path.join(temp_path, "stylesheets.html")) as f:
            info = f.read()
        txt = txt.replace("{{STYLESHEETS}}", info)
    txt = txt.replace("{{URL}}", url)
    txt = txt.replace("{{VERSION}}", basix.__version__)
    txt = re.sub(r"\{\{link:URL/([^\}]*)\}\}", link_markup, txt)
    return txt


# Make paths
if os.path.isdir(html_path):
    os.system(f"rm -r {html_path}")
os.system(f"mkdir {html_path}")
if os.path.isdir(temp_path):
    os.system(f"rm -r {temp_path}")
os.system(f"mkdir {temp_path}")

# Prepare templates
for file in ["stylesheets.html", "navbar.html", "intro.html", "outro.html"]:
    with open(os.path.join(template_path, file)) as f:
        content = f.read()
    with open(os.path.join(temp_path, file), "w") as f:
        f.write(insert_info(content))

for file in ["cpp/header.html", "python/source/_templates/layout.html"]:
    with open(os.path.join(path, f"{file}.template")) as f:
        content = f.read()
    with open(os.path.join(path, file), "w") as f:
        f.write(insert_info(content))

# Copy images and assets
os.system(f"cp -r {img_path} {html_path}/img")
os.system(f"cp -r {assets_path} {html_path}/assets")

# Convert markdown to html
for file in os.listdir(path):
    if file.endswith(".md"):
        with open(os.path.join(path, file)) as f:
            contents = unicode_to_html(insert_info(f.read()))
        with open(os.path.join(html_path, file[:-3] + ".html"), "w") as f:
            f.write(make_html_page(markdown.markdown(contents)))

# Make cpp docs
assert os.system(f"cd {path}/cpp && doxygen && cp -r {path}/cpp/html {html_path}/cpp") == 0

# Make python docs
assert os.system(f"cd {path}/python && make html && cp -r {path}/python/build/html {html_path}/python") == 0
