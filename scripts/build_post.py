import argparse

from utils import ROOT, build_post

ROOT_MAP = {
    "local": "../../",
    "github": "/from-first-principles/",
}

parser = argparse.ArgumentParser()

# Positional argument for post slug
parser.add_argument(
    "slug",
    help="Post folder name",
)

# Boolean flag for GitHub Pages build
# parser.add_argument(
#     "--github",
#     action="store_true",
#     help="Build for GitHub Pages",
# )

# Named argument for target build
parser.add_argument(
    "--target",
    choices=["local", "github"],
    default="local",
    help="Build for Local vs GitHub Pages",
)

args = parser.parse_args()

post_folder = ROOT / "posts" / args.slug

if not post_folder.exists():
    raise FileNotFoundError(f"No post called '{args.slug}'")

root = ROOT_MAP[args.target]

build_post(post_folder, root)