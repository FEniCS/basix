import markdown
import os

path = os.path.dirname(os.path.realpath(__file__))
html_path = os.path.join(path, "html")
img_path = os.path.join(path, "../img")
assets_path = os.path.join(path, "assets")


def unicode_to_html(txt):
    txt = txt.replace(u"\xe9", f"&eacute;")
    return txt


def make_html_page(html):
    with open(os.path.join(path, "intro.html")) as f:
        pre = f.read()
    with open(os.path.join(path, "outro.html")) as f:
        post = f.read()
    return pre + html + post

def insert_info(txt):
    if "{{SUPPORTED ELEMENTS}}" in txt:
        with open(os.path.join(path, "../README.md")) as f:
            info = "## Supported elements" + f.read().split("## Supported elements")[1].split("\n## ")[0]
        txt = txt.replace("{{SUPPORTED ELEMENTS}}", info)
    if "{{INSTALL}}" in txt:
        with open(os.path.join(path, "../INSTALL.md")) as f:
            info = f.read()
        txt = txt.replace("{{INSTALL}}", info)
    return txt


if os.path.isdir(html_path):
    os.system(f"rm -r {html_path}")
os.system(f"mkdir {html_path}")

os.system(f"cp -r {img_path} {html_path}/img")
os.system(f"cp -r {assets_path} {html_path}/assets")

for file in os.listdir(path):
    if file.endswith(".md"):
        with open(os.path.join(path, file)) as f:
            contents = unicode_to_html(insert_info(f.read()))
        with open(os.path.join(html_path, file[:-3] + ".html"), "w") as f:
            f.write(make_html_page(markdown.markdown(contents)))
