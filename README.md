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

## TODO
- [ ] For N NLI verification the hypothesis N group that score above threshold, should be group together and selecting all that as another row while each row still has info about its own group for eye-ball judging. (This one is for keeping valid responses and still discarding invalid ones )
- [X] Refine llm parser and use category based prompting.
- [X] Add NLI that capable of keeping valid numerical responses.
- [X] Graph filtered by topics
- [ ] Tracing better with Seismic 2d/3d data filtering and natural language.
- [ ] Build every verification tools for logging and tracing.