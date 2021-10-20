import argparse
import markdown
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

args = parser.parse_args()
url = args.url
url_no_http = url.split("//")[1]


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
            data = yaml.load(f)
        a = b
        for i in data:
            this_c = c
            for j, k in i.items():
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
        a = b + "<h2 id=\"project_subtitle\">Documentation</h2>" + c

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
    if "{{SUPPORTED ELEMENTS}}" in txt:
        with open(path("../../README.md")) as f:
            info = "## Supported elements" + f.read().split("## Supported elements")[1].split("\n## ")[0]
        txt = txt.replace("{{SUPPORTED ELEMENTS}}", info)
    if "{{INSTALL}}" in txt:
        with open(path("../../INSTALL.md")) as f:
            info = f.read()
        txt = txt.replace("{{INSTALL}}", info)
    txt = txt.replace("{{URL}}", url)
    txt = txt.replace("{{VERSION}}", basix.__version__)
    txt = re.sub(r"\{\{link:URL/([^\}]*)\}\}", link_markup, txt)
    return txt


# Make paths
for dir in ["html", "cpp", "python", "wweb"]:
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
# os.system(f"git clone http://github.com/FEniCS/web {path('web')}")

with open(path('web/_layouts/default.html')) as f:
    intro, outro = f.read().split("\n    {{ title }}\n    {{ content }}\n")
    intro = insert_info(jekyll(intro))
    outro = insert_info(jekyll(outro))

with open(path("template/navbar.html")) as f:
    intro += f"<h2 id=\"project_subtitle\">{insert_info(f.read())}</h2>"

for file in ["stylesheets.html", "navbar.html"]:
    with open(os.path.join(path('template'), file)) as f:
        content = f.read()
    with open(os.path.join(temp, file), "w") as f:
        f.write(insert_info(content))

with open(os.path.join(temp, "intro.html"), "w") as f:
    f.write(intro.replace("{{SEO}}", "<title>Basix documentation</title>"))
with open(os.path.join(temp, "outro.html"), "w") as f:
    f.write(outro)

with open(path("cpp/footer.html"), "w") as f:
    f.write(outro)
with open(path("cpp/header.html"), "w") as f:
    with open(path("cpp-seo.html")) as f2:
        f.write(intro.replace("{{SEO}}", f2.read()))
    with open(path("cpp-header.html")) as f2:
        f.write(f2.read())


for fin, fout in [
    ("python-layout.html.template", "python/source/_templates/layout.html")
]:
    with open(path(fin)) as f:
        content = f.read()
    with open(path(fout), "w") as f:
        content = insert_info(content)
        f.write(insert_info(content))

with open("cpp/Doxyfile") as f:
    content = ""
    for line in f:
        if line.startswith("HTML_HEADER"):
            content += f"{line.split('=')[0]}= header.html\n"
        elif line.startswith("HTML_FOOTER"):
            content += f"{line.split('=')[0]}= footer.html\n"
        else:
            content += line.replace(" ..", " ../..")

with open("cpp/Doxyfile", "w") as f:
    f.write(content)

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
