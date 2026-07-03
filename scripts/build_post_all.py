import argparse

from utils import ROOT, build_post


ROOT_MAP = {
    "local": "../..",
    "github": "/from-first-principles",
}


parser = argparse.ArgumentParser()

parser.add_argument(
    "--target",
    choices=["local", "github"],
    default="local",
    help="Build target",
)

args = parser.parse_args()

root = ROOT_MAP[args.target]


POSTS = ROOT / "posts"

built = 0
skipped = 0

for post_dir in POSTS.iterdir():

    if not post_dir.is_dir():
        continue

    md_path = post_dir / "post.md"

    if not md_path.exists():
        continue

    try:

        build_post(post_dir, root)

        built += 1

    except Exception as e:

        print(f"⚠ Skipping '{post_dir.name}': {e}")

        skipped += 1


print()
print("=" * 60)
print(f"Built   : {built}")
print(f"Skipped : {skipped}")
print("=" * 60)