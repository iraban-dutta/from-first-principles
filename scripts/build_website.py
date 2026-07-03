from pathlib import Path
import argparse
import subprocess
import sys


ROOT = Path(__file__).resolve().parent.parent


def run(script_name, target):

    script = ROOT / "scripts" / script_name

    print(f"\n{'=' * 60}")
    print(f"Running {script_name} ({target})")
    print(f"{'=' * 60}\n")

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--target",
            target,
        ],
        cwd=ROOT,
    )

    if result.returncode != 0:
        raise RuntimeError(f"{script_name} failed.")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--target",
        choices=["local", "github"],
        default="local",
        help="Build target",
    )

    args = parser.parse_args()

    print("\nBuilding website...\n")
    print(f"Target: {args.target}")

    run("build_post_all.py", args.target)
    run("build_pages.py", args.target)

    print()
    print("=" * 60)
    print(f"✓ Website built successfully ({args.target})")
    print("=" * 60)


if __name__ == "__main__":
    main()