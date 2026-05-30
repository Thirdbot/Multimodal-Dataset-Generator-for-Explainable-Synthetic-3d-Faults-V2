# Installation
```bash
git clone ...
cd  repo
git submodule update --init --recursive 
uv python install 3.11
uv venv --python 3.11
uv run python scripts/sync_synthoseis_deps.py
uv sync --group synthoseis
# test environment
uv run --group synthoseis python --version
uv run --group synthoseis python -c "import numpy, zarr; print('deps ok')"
```

