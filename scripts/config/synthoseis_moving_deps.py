"""Copy Synthoseis dependency declarations into this project's uv group.

This is a maintenance helper for syncing third_party/synthoseis dependencies
into the local [dependency-groups].synthoseis lock workflow.
"""

import tomllib
import subprocess
from pathlib import Path

p = Path("third_party/synthoseis/pyproject.toml")
data = tomllib.loads(p.read_text())
deps = data["project"].get("dependencies", [])

print("Adding Synthoseis dependencies to group [dependency-groups].synthoseis")
for dep in deps:
    print("  ", dep)

subprocess.run(["uv", "add", "--group", "synthoseis", *deps], check=True)
