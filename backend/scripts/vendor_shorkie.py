"""Re-sync the vendored Shorkie network definition from a local checkout.

    uv run python scripts/vendor_shorkie.py --src /path/to/shorkie-pytorch [--check]

`--check` exits non-zero when the vendored copy has drifted, so it can gate a release.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TARGET = HERE.parent / "app" / "models" / "_shorkie_net.py"
DEFAULT_SRC = Path("/Users/tds122/Documents/shorkie-pytorch")

HEADER = '''"""Vendored from github.com/tdsone/shorkie-pytorch @ {sha}

Do not edit here. Re-sync with: uv run python scripts/vendor_shorkie.py

Vendored rather than pip-installed because upstream cannot currently be installed:
its pyproject declares no [build-system] and no py-modules, so the built wheel holds
only dist-info metadata (pip install succeeds and `import shorkie` then fails), and
its requires-python of >=3.13 would exclude the 3.12 deployment image regardless.
"""

'''


def render(src: Path) -> str:
    sha = subprocess.run(
        ["git", "-C", str(src), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    body = (src / "shorkie.py").read_text()
    return HEADER.format(sha=sha) + body


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    if not (args.src / "shorkie.py").exists():
        raise SystemExit(f"no shorkie.py under {args.src}")

    rendered = render(args.src)
    if args.check:
        current = TARGET.read_text() if TARGET.exists() else ""
        if current != rendered:
            print("vendored copy is out of date; re-run without --check", file=sys.stderr)
            return 1
        print("vendored copy is current")
        return 0

    TARGET.write_text(rendered)
    print(f"wrote {TARGET.relative_to(HERE.parent)} ({len(rendered.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
