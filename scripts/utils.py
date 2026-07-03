from pathlib import Path
import yaml
import markdown
import re

ROOT = Path(__file__).resolve().parent.parent

TEMPLATES = ROOT / "templates"
HEADER = (TEMPLATES / "header.html").read_text(encoding="utf-8")
FOOTER = (TEMPLATES / "footer.html").read_text(encoding="utf-8")
POST_TEMPLATE = (TEMPLATES / "post.html").read_text(encoding="utf-8")
PAGE_TEMPLATE = (TEMPLATES / "page.html").read_text(encoding="utf-8")


def parse_markdown(md_path: Path):
    """
    Parse a markdown file with YAML front matter.
    """

    text = md_path.read_text(encoding="utf-8")

    if not text.startswith("---"):
        raise RuntimeError(f"{md_path} missing metadata block.")

    _, metadata_text, body = text.split("---", 2)

    metadata = yaml.safe_load(metadata_text)

    return metadata, body.strip()


def build_toc(md: str):
    """
    Build a nested Table of Contents from H1/H2 headings.

    - First H1 (page title) is ignored.
    - H1 -> major bullet
    - H2 -> nested bullets
    - Inserts ids into headings.
    """

    processed = []

    toc = [
        '<details class="toc" id="table-of-contents">',
        "<summary>Table of Contents</summary>",
        "<ul>"
    ]

    first_h1 = True
    inside_h1 = False

    for line in md.splitlines():

        # ----------------------------------------------------
        # H1
        # ----------------------------------------------------
        if line.startswith("# "):

            title = line[2:].strip()

            # Skip article title
            if first_h1:
                first_h1 = False
                processed.append(line)
                continue

            slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")

            # close previous H1 block
            if inside_h1:
                toc.append("</ul>")
                toc.append("</li>")

            toc.append(
                f'<li><a href="#{slug}">{title}</a>'
            )
            toc.append("<ul>")

            inside_h1 = True

            processed.append(
                f'# <span id="{slug}"></span>{title}'
            )

        # ----------------------------------------------------
        # H2
        # ----------------------------------------------------
        elif line.startswith("## "):

            title = line[3:].strip()

            slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")

            processed.append(
                f'## <span id="{slug}"></span>{title}'
            )

            toc.append(
                f'<li><a href="#{slug}">{title}</a></li>'
            )

        else:
            processed.append(line)

    # Close the last nested list if we opened one
    if inside_h1:
        toc.append("</ul>")
        toc.append("</li>")

    # Close outer list
    toc.append("</ul>")

    # Close <details>
    toc.append("</details>")

    toc_html = "\n".join(toc)

    return "\n".join(processed), toc_html



def markdown_to_html(md: str):
    """
    Convert markdown to HTML.
    """

    return markdown.markdown(
        md,
        extensions=[
            "fenced_code",
            "tables",
            "attr_list",
        ],
    )


def build_post(post_folder: Path, root):
    """
    Build a single post.
    """

    md_path = post_folder / "post.md"

    metadata, body = parse_markdown(md_path)

    body, toc = build_toc(body)
    body = body.replace("{{TOC}}", toc)
    html = markdown_to_html(body)

    page = POST_TEMPLATE.replace("{{ROOT}}",root,)
    page = page.replace("{{title}}", metadata["title"])
    page = page.replace("{{header}}", HEADER.replace("{{ROOT}}", root))
    page = page.replace("{{footer}}", FOOTER.replace("{{ROOT}}", root))
    page = page.replace("{{content}}", html)

    output = post_folder / "index.html"

    output.write_text(page, encoding="utf-8")

    print(f"✓ Built {post_folder.name}")