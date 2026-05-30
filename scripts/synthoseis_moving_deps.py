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
