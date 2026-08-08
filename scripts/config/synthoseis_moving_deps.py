"""Copy Synthoseis dependency declarations into this project's uv group.

This is a maintenance helper for syncing third_party/synthoseis dependencies
into the local [dependency-groups].synthoseis lock workflow.
"""

import tomllib
import subprocess
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[2]
    p = root / "third_party" / "synthoseis" / "pyproject.toml"
    data = tomllib.loads(p.read_text())
    deps = data.get("project", {}).get("dependencies", [])

    print("Adding Synthoseis dependencies to group [dependency-groups].synthoseis")
    for dep in deps:
        print("  ", dep)

    if deps:
        subprocess.run(["uv", "add", "--group", "synthoseis", *deps], check=True)
    else:
        print("No dependencies found; skipping uv add.")


# Run the `uv add` only when invoked as a script (README: `python -m scripts.config.synthoseis_moving_deps`),
# never as an import side-effect -- importing this module used to mutate pyproject.toml/uv.lock.
if __name__ == "__main__":
    main()
