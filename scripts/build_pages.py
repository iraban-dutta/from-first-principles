import argparse
from pathlib import Path
from datetime import datetime, date
from collections import defaultdict

from utils import (
    ROOT,
    TEMPLATES,
    HEADER,
    FOOTER,
    PAGE_TEMPLATE,
    parse_markdown,
)

ROOT_MAP = {
    "local": "",
    "github": "/from-first-principles",
}

POSTS_DIR = ROOT / "posts"

# ============================================================
# Helpers
# ============================================================

MONTHS = [
    "",
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]


def pretty_date(value):
    """
    Convert either

    - datetime.date
    - datetime.datetime
    - YYYY-MM-DD string

    into

    3 July 2026
    """

    if isinstance(value, str):
        value = datetime.strptime(value, "%Y-%m-%d").date()

    elif isinstance(value, datetime):
        value = value.date()

    elif not isinstance(value, date):
        raise TypeError(
            f"Unsupported date type: {type(value)}"
        )

    return f"{value.day} {MONTHS[value.month]} {value.year}"


def load_posts():

    posts = []

    for folder in POSTS_DIR.iterdir():

        if not folder.is_dir():
            continue

        md = folder / "post.md"

        if not md.exists():
            continue

        try:

            metadata, _ = parse_markdown(md)

            posts.append(metadata)

        except Exception as e:

            print(f"⚠ Skipping '{folder.name}': {e}")

            continue

    posts.sort(
        key=lambda p: p["published"],
        reverse=True,
    )

    return posts


def render_page(title: str, body: str, root: str):

    page = PAGE_TEMPLATE

    page = page.replace("{{ROOT}}", root)
    page = page.replace("{{title}}", title)
    page = page.replace("{{header}}", HEADER.replace("{{ROOT}}", root))
    page = page.replace("{{footer}}", FOOTER.replace("{{ROOT}}", root))
    page = page.replace("{{content}}", body)

    return page


# ============================================================
# Home
# ============================================================

def build_home(posts, root):

    html = []

    html.append("""
<section class="tagline">

<h1>From First Principles</h1>

<p>
Hi, this is Iraban.
Below are my learning notes on topics in AI, ML, Maths and CS.
</p>

</section>
""")

    for i, post in enumerate(posts):

        html.append(f"""
<div class="post">

<div class="date">
{pretty_date(post["published"])}
</div>

<h2>

<a href="{root}/posts/{post['slug']}/">

{post['title']}

</a>

</h2>

<p>

{post.get("description", "")}

</p>

</div>
""")

        if i != len(posts) - 1:
            html.append("<hr>")

    page = render_page(
        "From First Principles",
        "\n".join(html),
        root,
    )

    (ROOT / "index.html").write_text(
        page,
        encoding="utf-8",
    )

    print("✓ Built index.html")


# ============================================================
# Archive
# ============================================================

def build_archive(posts, root):

    grouped = defaultdict(lambda: defaultdict(list))

    for post in posts:

        dt = post["published"]

        grouped[dt.year][dt.month].append(post)

    html = ["<h1>Archive</h1>"]

    for year in sorted(grouped.keys(), reverse=True):

        html.append(f"<h2>{year}</h2>")

        for month in sorted(grouped[year].keys(), reverse=True):

            html.append(f"<h3>{MONTHS[month]}</h3>")
            html.append("<ul>")

            for post in grouped[year][month]:

                html.append(f"""
<li>
<a href="{root}/posts/{post['slug']}/">
{post['title']}
</a>
</li>
""")

            html.append("</ul>")

    page = render_page(
        "Archive | From First Principles",
        "\n".join(html),
        root,
    )

    (ROOT / "archive.html").write_text(
        page,
        encoding="utf-8",
    )

    print("✓ Built archive.html")


# ============================================================
# Static Pages
# ============================================================

def build_static_page(filename, title, root):

    content = (
        TEMPLATES / filename
    ).read_text(encoding="utf-8")

    content = content.replace("{{ROOT}}", root)

    page = render_page(
        title,
        content,
        root,
    )

    (ROOT / filename).write_text(
        page,
        encoding="utf-8",
    )

    print(f"✓ Built {filename}")


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--target",
        choices=["local", "github"],
        default="local",
        help="Build target",
    )

    args = parser.parse_args()

    root = ROOT_MAP[args.target]

    posts = load_posts()

    build_home(posts, root)
    build_archive(posts, root)

    build_static_page(
        "about.html",
        "About | From First Principles",
        root,
    )

    build_static_page(
        "search.html",
        "Search | From First Principles",
        root,
    )

    print()
    print("=" * 60)
    print(f"Built pages for target: {args.target}")
    print("=" * 60)


if __name__ == "__main__":
    main()