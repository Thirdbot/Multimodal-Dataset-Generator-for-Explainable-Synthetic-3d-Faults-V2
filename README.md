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
- [ ] Each sample recipe will add another data from allowing category such as Fault,Closure,etc. Sample must have at least 1 allow category.
- [ ] Refine llm parser and use category based prompting.
- [ ] Add NLI that capable of keeping valid numerical responses.
- [ ] Improve Tracing and Graph Generation for better understanding, right now it tracing by topics like closures,fault.
- [ ] Build every verification tools for logging and tracing.