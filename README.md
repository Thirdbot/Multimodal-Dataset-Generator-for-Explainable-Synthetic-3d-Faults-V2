# Multimodal Synthetic Seismic Dataset Generator

Generates a grounded VQA / referring-segmentation dataset from **synthetic** 3D seismic:
Synthoseis physics build → property-graph extraction → 2D view-images + segmentation masks →
a local LLM writes question/answer pairs → NLI + retrieval gates verify them → a multimodal CSV
you can push to Hugging Face. Because the scenes are synthetic, every label (throw, dip, area,
count, intersection) is exact ground truth from the generator.

---

# Installation
```bash
git clone ...
cd  repo
git submodule update --init --recursive 
uv python install 3.11
uv venv --python 3.11
uv run python -m scripts.config.synthoseis_moving_deps
uv sync --group synthoseis
# test environment
uv run --group synthoseis python --version
uv run --group synthoseis python -c "import numpy, zarr; print('deps ok')"
```

---

# End-to-end user guide

## Pipeline at a glance

```
 control_parameter.py          settings.yaml
   (what & how many)   ─┐        (physics/scene)
                        ▼
   recipes/  →  build_configs/  →  sample_generator (Synthoseis)  →  builds/<scene>/parameters.db + object .zarr
                                                                          │  trace (graph_generator → graph_system)
                                                                          ▼
                                          graphs/<scene>_db_extract.json  →  graphs/properties_graph/…_properties_graph.json
                                                                          │  image + 2d-graph (images_generator, compute_attribute)
                                                                          ▼
        build_objects/images/<scene>/  (2D views + masks)   +   graphs/properties_2d_graph/…_properties_2d_graph.json
                                                                          │  QA (generator_pipeline: LLM writes Q/A, NLI+retrieval verify)
                                                                          ▼
                          Dataset/verified_qa_shard_*.jsonl  →  merge/balance  →  Dataset/verified_qa.jsonl
                                                                          │  DatasetMaker
                                                                          ▼
                                       Dataset/multimodal_multi_image_dataset.csv   →  Hugging Face
```

The whole run is driven by `scripts/run_all.sh`; you rarely call the stages by hand.

---

## 1. Where to configure

### 1a. Physics & scene size — `settings.yaml`
The Synthoseis build parameters and the output paths. Most-changed:

| key | meaning |
|---|---|
| `control.cube_shape` | voxel dims `[x, y, z]` — bigger = more detail, much slower/heavier |
| `control.incident_angles`, `control.digi`, `control.signal_to_noise_ratio_db` | acquisition / imaging realism |
| `control.min_number_faults` / `max_number_faults` | global fault-count bounds (per-category overrides live in `control_parameter.py`) |
| `control.closure_types`, `include_salt`, `basin_floor_fans` | which geobodies can appear |
| `control.min_closure_voxels_*`, `max_column_height` | closure sizing thresholds |
| `paths.*` | where builds/graphs/configs land (defaults are fine) |

### 1b. What to generate — `scripts/config/control_parameter.py`
The **class mix and count** live in the settings dicts near the bottom (`sample_types`,
`ratio_per_types`, and the population amount):

- `sample_types` — which scene categories to draw from. Available: `boring`, `fault_only`,
  `fault_complex`, `salt_only`, `salt_fault_mixed`, `onlap`, `depositional`, `full_mixed`
  (each is a method on `CategoricalParameter`; `boring` = featureless → empty-mask negatives).
- `ratio_per_types` — the target proportion of each class (e.g. `fault_only: 0.45`). Set a class
  to `0.0` (or omit it) to exclude it.
- Per-category fault ranges are the method defaults, e.g. `fault_only(f_min=1, f_max=9)`,
  `fault_complex(f_min=10, f_max=20)` — edit those methods to change density.
- Scene **resolution presets** `LOW` / `MEDIUM` / `HIGH` are defined here; only `LOW` is wired up
  today (see the `__main__` block) — point `cube_shape` at another preset to change it.

You normally don't run this file directly — the build driver populates recipes/configs from it.

### 1c. Run knobs — environment variables
Pass these on the `run_all.sh` command line (all have sane defaults):

| var | stage | default | what it does |
|---|---|---|---|
| `TARGET_SCENES` | build | 400 | how many scenes to build |
| `BUILD_CONCURRENCY` | build | 6 | parallel Synthoseis builds (CPU + RAM heavy) |
| `TRACE_CONCURRENCY` | build | 1 | parallel property-graph extractions |
| `IMAGE_GEN_CONCURRENCY` | build | 3 | parallel image/2d-graph subprocesses |
| `BUILD_TIMEOUT` | build | 900 | seconds before a hung build is killed + quarantined |
| `MIN_FREE_GB` / `MAX_RAW_BACKLOG` / `QUEUE_AHEAD` | build | 20 / 4 / 9 | disk floor + backpressure |
| `RECONCILE_INTERVAL` | build | 45 | self-heal scan cadence (lower = tighter, more CPU) |
| `N_SHARDS` | qa | 8 | parallel QA workers |
| `NLI_DEVICE` / `EMBED_DEVICE` | qa | cuda | put NLI/embeddings on `cpu` if the GPU is busy |
| `OBJECTS_PER_SEED` | qa | 3 | objects grouped per question seed |
| `DATASET_REGEN` | qa | – | `1` = truncate + regenerate all QA (after a prompt change) |
| `LLM_ENDPOINT` | qa | `http://localhost:8000/v1` | where the LLM server is |
| `OVERLAP_BUILD_CONCURRENCY`, `OVERLAP_N_SHARDS`, `OVERLAP_MIN_BATCH` | overlap | 3 / 4 / 24 | leaner values used when build+QA run together |

**Geological "reading" thresholds** (turn raw values into phrases like "major throw"), in
`NaturalTransform/text_transform.py`, also env-overridable: `THROW_MINOR_MS` (40), `THROW_MAJOR_MS`
(120), `AREA_SMALL_PCT` (2), `AREA_BROAD_PCT` (8).

**LLM server knobs** (see `scripts/llm_server.sh`): `LLM_MODEL`, `LLM_PORT`, `LLM_CONTEXT`,
`LLM_MEM_FRAC`, `LLM_QUANT`, `LLM_EXTRA_ARGS`.

### 1d. The Q&A prompt
`Verifier/llm_machine.py` → `MASTER_PROMPT` is the system prompt that steers question/answer
style, coordinate/bbox referring, and de-biasing. The model id defaults to
`Qwen/Qwen2.5-1.5B-Instruct` (override with `LLM_MODEL`).

---

## 2. How to generate the dataset

**Step 1 — start the local LLM server** (one-time setup, then serve; leave it running):
```bash
bash scripts/llm_server.sh setup sglang     # one-time: builds an isolated .venv-serve
```
```bash
LLM_MODEL=Qwen/Qwen2.5-1.5B-Instruct bash scripts/llm_server.sh sglang
```
> Check `nvidia-smi` first — if a training job owns the GPU, either lower `LLM_MEM_FRAC` or run QA
> with `NLI_DEVICE=cpu`. `run_all.sh` only *checks* the endpoint is up; it never starts the server.

**Step 2 — run the pipeline** (in another terminal):
```bash
TARGET_SCENES=400 N_SHARDS=8 bash scripts/run_all.sh all
```
Modes (`scripts/run_all.sh <mode>`):
- `all` — build → QA → finalize, **sequentially** (each phase gets the whole machine).
- `overlap` — build and QA run **at the same time** (GPU isn't idle during the build).
- `build` — build only (CPU; no GPU/LLM needed). Do this when the GPU is busy, QA later.
- `qa` — QA over whatever graphs exist (resume-safe; re-runnable).
- `finalize` — merge shards → balance classes → build the CSV.

**What lands where:**
- `build_objects/images/<scene>/` — the 2D view-images and segmentation masks
- `build_objects/objects/` — the raw 3D object arrays (`.zarr`)
- `graphs/<scene>_db_extract.json`, `graphs/properties_graph/`, `graphs/properties_2d_graph/`
- `Dataset/verified_qa_shard_*.jsonl` → merged/balanced `Dataset/verified_qa.jsonl`
- **`Dataset/multimodal_multi_image_dataset.csv`** — the final dataset (image paths + masks + Q/A + evidence)

Progress helpers: `scripts/gen_monitor.sh`, `scripts/qa_monitor.sh`.

---

## 3. How to add a new attribute

An "attribute" is a fact attached to an object node that the Q&A can then ask/answer about.
There are two sources, and one optional phrasing step.

### 3a. Attribute that exists in the physics DB
The extractor copies a whitelisted set of `parameters.db` columns onto graph nodes. Add the column
name to the right list in **`scripts/graph/graph_generator.py`**:
- `MODEL_KEYS` — per-scene / model-level fields
- `FAULT_KEYS` — per-fault fields (e.g. throw, azimuth)
- `CLOSURE_KEYS` — per-closure fields (e.g. fluid, volume)

`CATEGORY_FILTERS` (same file) maps each scene category → which of those key sets apply. Add your
key, rebuild the graphs (`scripts/clean.sh rebuild --yes` then regenerate, or just re-run trace on
new scenes) and it shows up as a node attribute.

### 3b. Attribute computed from the 2D mask (geometry)
Mask geometry (area, dip, …) is computed in **`scripts/graph/compute_attribute.py`**:
1. Write a function `my_attr(mask) -> value` alongside `area_pct` / `dip_degrees` (they show the
   `np.argwhere` row/col↔y/x convention and how to guard empty masks).
2. In `mask_features(...)`, add `features["my_attr"] = my_attr(mask)`.
3. Add `"my_attr"` to `MASK_FEATURE_KEYS`.

`scripts/graph/properties_2d_graph.py` already reads `MASK_FEATURE_KEYS` and attaches every key to
the per-view 2D-graph nodes — no change needed there.

### 3c. Make it appear in the Q&A text (optional)
The raw value is usable as-is, but to phrase it naturally (a value sentence or a geological
"reading"), edit **`NaturalTransform/text_transform.py`** — `relations_to_evidence` /
`_reading_sentences`. Add a branch for your `trace_type`/edge, and (if it's a thresholded reading)
a knob like the existing `THROW_*` / `AREA_*` env thresholds so the cutoff is tunable and grounded.

> After changing what facts exist, regenerate QA with `DATASET_REGEN=1` so old rows are rebuilt.

---

## 4. How to upload to Hugging Face

The final CSV (`Dataset/multimodal_multi_image_dataset.csv`) is packaged and pushed as a HF dataset.

```bash
huggingface-cli login          # or: export HF_TOKEN=hf_xxx
uv run python -m Dataset.upload_to_huggingface
```
The target repo id is set in the `__main__` block of `Dataset/upload_to_huggingface.py`
(currently `thirdExec/synthetic-seismic-vlm`, `private=False`) — edit that line to point at your
own repo, or call `upload_dataset(DEFAULT_CSV, "you/your-dataset", private=True, dry_run=True)` to
preview first. It resolves image/mask paths, parses the evidence/regions JSON, creates the repo if
needed, and `push_to_hub`s.

---

## 5. How to clean up

`scripts/clean.sh` is **dry-run by default** (prints what it would remove, with sizes); add
`--yes` to actually delete. It **never** removes `Dataset/*.csv`.

```bash
scripts/clean.sh after           # post-run tidy: build_configs recipes builds graphs/_shard_*
                                 #   (keeps scenes, graphs, Dataset)
scripts/clean.sh qa              # redo QA only: drops verified_qa*.jsonl + graph shards (keeps scenes/graphs)
scripts/clean.sh rebuild         # full reset to 0 scenes: removes build_objects/{images,objects},
                                 #   graphs/{properties_graph,properties_2d_graph,_shard_*,*_db_extract.json},
                                 #   build_configs, recipes, builds  (keeps Dataset/)
```
Add `--yes` to any of them to execute, e.g. `scripts/clean.sh rebuild --yes`.
> After `rebuild`, the old `Dataset/*.csv` will reference deleted images — regenerate or clear it.

---

## TODO
- [ ] For N NLI verification the hypothesis N group that score above threshold, should be group together and selecting all that as another row while each row still has info about its own group for eye-ball judging. (This one is for keeping valid responses and still discarding invalid ones )
- [X] Refine llm parser and use category based prompting.
- [X] Add NLI that capable of keeping valid numerical responses.
- [X] Graph filtered by topics
- [ ] Tracing better with Seismic 2d/3d data filtering and natural language.
- [ ] Build every verification tools for logging and tracing.
