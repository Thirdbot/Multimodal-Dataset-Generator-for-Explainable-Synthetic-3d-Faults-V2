# simulationv2 — Project Overview

A local, end-to-end generator for a **multimodal seismic-interpretation dataset**. It synthesizes 3D seismic volumes with a physics-based simulator, turns each volume's ground-truth metadata into a knowledge graph, renders 2D sections + segmentation masks, and uses a graph-RAG + LLM + NLI loop to write **grounded, verified visual question–answer pairs** with pixel masks — the kind of data you fine-tune a seismic VLM on.

The whole thing runs on one machine, orchestrated by a set of asyncio file-watchers so that each finished sample flows through the pipeline independently instead of waiting for the whole batch.

---

## 1. Project goal

Train and evaluate **vision-language models on seismic interpretation** (fault detection, closure/salt identification, counting, localization, geometry). Real labelled seismic is scarce, proprietary, and expensive to annotate. This project manufactures an arbitrarily large, fully-labelled corpus where **every answer is traceable to a known simulation parameter or a measured pixel region** — no human annotation, no hallucinated geology.

Each dataset row is:

- a **seismic section image** (grayscale 2D slice of the shared scene),
- **one or more segmentation masks** — one per region the question/evidence touches (object masks, plus a union mask for whole-section facts),
- a natural **question** that references objects **by coordinate** ("the throw of the fault at [57.5,289]?"),
- a **verified answer** in plain natural language (no tags),
- structured **evidence** — `<evidence>` with one plain `<region>…<SEG></region>` block per mask (tags unwrapped, value words kept so it's retrievable),
- structured **regions** — one per mask/`<SEG>`: `object_name` (coord reference, or `"the section"` for a global region), `class_id`, `bbox`, `center`, and the regressable `values`.

The non-negotiable design property: **truthfulness**. Answers are grounded in `parameters.db` values and mask-measured geometry, gated by **NLI coverage** (every fact entailed, answer gate trust ≥ 0.80), a question-coverage **edge gate**, topic/count/attribute guards, and a coordinate-based entity-swap guard, so the model learns real correspondences instead of plausible-sounding fiction.

> **Removed:** an earlier `reason` (chain-of-thought) column. It was generated *after* verification, so it was the one un-checked field in a dataset whose whole premise is verifiability. It is gone from the pipeline and the schema.

---

## 2. Why Synthoseis

[Synthoseis](https://github.com/sede-open/synthoseis) (vendored under `third_party/synthoseis`) is an open, physics-based synthetic seismic generator. It's used because it gives us the one thing real data can't: **perfect, complete ground truth for free**.

- **Labels are exact and exhaustive.** Every fault, closure, salt body, onlap, lithology and age-depth volume is known at voxel resolution, and their parameters (fault throw, tilt, fluid type, closure size, intersections…) are dumped to a per-build SQLite `parameters.db`. There's nothing to annotate — the simulator already knows.
- **Controllable difficulty & composition.** We dial scenes from `boring` (stratigraphy only) up to `fault_complex` (10–20 faults) or `full_mixed` (faults + salt + onlap), so the dataset can be balanced across structural regimes.
- **Physical plausibility.** Output is wavelet-convolved reflectivity with realistic bandwidth and noise, so sections look like genuine seismic rather than cartoons — the VLM sees the same textures it will face in practice.
- **3D → many 2D.** One volume yields inline / crossline / timeslice sections and many object subsets, amplifying grounding coverage per expensive build.

The trade-off (a known limitation, see §12): it's a **single simulator**, so the visual distribution has a ceiling. Synthoseis is the label oracle, not a substitute for real-data fine-tuning at the end.

### The guarded build wrapper

Synthoseis has internal fault modes that ignore small `max_number_faults` and doesn't natively emit one mask per fault. `scripts/build/synthoseis_config_guard.py` wraps `build_model` with two temporary monkeypatches (reverted in a `finally`):

- **`_fault_settings_override`** — for `no_faults` categories it zeroes throw/mode/count and sets `fmode="none"`; for low-count categories (`max_faults < 6`) it forces `mode="random"` so the JSON range wins instead of a Synthoseis preset.
- **`_individual_fault_output_override`** — patches `Faults.get_displacement_vector` so that, when `model_qc_volumes` is on, each fault's thresholded mask (`fault_segm > 0.25 & displacement_classification > 1.0`) is written to `faults/fault_XX.zarr` **before** it's merged into the cumulative fault volume. This is what makes **per-fault masks** possible downstream — without it we'd only have one merged all-faults blob.

Because these patches mutate process-global Synthoseis state, **each build runs in its own subprocess** (see §11), so the patches can never leak into the watcher process.

---

## 3. How generation works (recipes → configs → builds)

**Stage owner:** `scripts/config/control_parameter.py`, `scripts/build/sample_generator.py`

### 3.1 Settings → parameter template
`settings.yaml` holds the low-level Synthoseis knobs (cube shape, sampling `digi`, `infill_factor`, layer thickness, closure voxel minimums, S/N ratio, bandwidth, QC flags…). `CategoricalParameter` loads them into a mutable template and exposes per-category presets:

| Category | Salt | Fans | Faults | Closure types |
|---|---|---|---|---|
| `boring` | – | – | 0 | simple |
| `fault_only` | – | – | 1–9 | faulted |
| `fault_complex` | – | – | 10–20 | faulted |
| `salt_only` | ✓ | – | 0 | simple |
| `salt_fault_mixed` | ✓ | – | 1–4 | faulted+simple |
| `onlap` | – | – | 0 | onlap |
| `depositional` | – | ✓ | 0 | simple+onlap |
| `full_mixed` | ✓ | ✓ | 2–6 | simple+faulted+onlap |

### 3.2 Population → recipe + build configs
`high_level_controls` declares `sample_population` (how many samples) and `sample_types` / `ratio_per_types` (the category mix; unspecified types split the remaining ratio evenly). `SampleControl.populate()`:

1. draws each sample's category by **weighted-random** (`random.choices(categories, weights=ratio_per_types)` per sample), which converges to the target ratios across many small recipes — replacing the earlier per-category integer counting that **floored small-population ratios to 0** and over-filled the first categories (the bug that made an early balanced run come out all-`fault_complex`),
2. writes one JSON **build config** per sample to `build_configs/{category}_{uuid}.json`,
3. writes a **recipe** `recipes/recipe_N.yaml` listing the sample names + the ratio plan.

**Current setting — balanced-yet-realistic mix.** All eight categories are enabled, with `ratio_per_types` weighting toward what's geologically common: `fault_complex 0.22, salt_fault_mixed 0.15, fault_only 0.13, onlap 0.12, salt_only 0.10, depositional 0.10, full_mixed 0.10, boring 0.08` (sums to 1.0). Resulting **type presence** across scenes: fault ~0.60, closure ~0.87, salt ~0.35, onlap ~0.32 — every type well-represented (balance), faults/closures dominant (realism), none forced to parity. `sample_population: 4` per recipe, deliberately small; scale comes from **calling `populate()` repeatedly** (§11.1), not one huge recipe.

**Seismic realism (`seismic_signal_controls`).** `signal_to_noise_ratio_db = [6, 12, 18]` — a triangular `[min, mode, max]` in dB sampled per scene, so sections span *noisy → clean* around a legible centre; `bandwidth_low [3,6]` / `bandwidth_high [20,35]` Hz / `ord 4` set the wavelet band (vertical resolution). Coherent noise (migration smiles/frowns) is flipped in per scene by Synthoseis. **Crucially, noise only varies the seismic *image* — masks, counts, dip, throw, area all come from the geometry volumes, so the labels stay exact.** Noise makes the task realistic without corrupting ground truth.

### 3.3 Build execution
`BuildGenerator.build_sample()` runs one config through `guarded_build_model`. On success it appends the build folder to `builds/success.yaml`; on failure to `builds/failed.yaml`. Both trackers are written **atomically** (temp file + `os.replace`) under a **cross-process `fcntl.flock`** (`builds/.tracker.lock`), so parallel build subprocesses never lose an entry and the watchers never read a torn file. A success also drops the config from `failed.yaml` so a later failed-list rewrite can't `rmtree` a now-good build. `build_sample` is also exposed as a CLI (`python -m scripts.build.sample_generator --config … --run-id …`) so the watcher can launch each build as an isolated process.

Output per build: a `builds/seismic__DDDD_DDDD_{category}_{uuid}/` folder containing `parameters.db`, the seismic zarr cubes, per-object zarr volumes (faults, closures by fluid, salt, onlap, lithology, age/depth), and (thanks to the guard) `faults/fault_XX.zarr`.

---

## 4. Properties extraction from `parameters.db`

**Stage owner:** `scripts/graph/low_level_tracer.py` (`ParameterDbTracer`)

This is the **low-level, non-interpretive** layer. It opens the build's SQLite `parameters.db`, lists every table, `SELECT *`s each one verbatim, and dumps them to `graphs/{sample}_db_extract.json`. It does not decide what matters — it just preserves rows by table (`model_parameters`, `fault_parameters`, `closure_parameters`, …) so the next stage can filter.

Key raw fields that survive here (used later):

- **model_parameters:** `number_faults`, `fault_mode`, `salt_inserted`, `number_hc_closures`, `number_fault_intersections`, `fault_voxel_count_list`, `infill_factor`.
- **fault_parameters (per fault):** `throw`, `tilt_pct`, plus the ellipsoid geometry (`a,b,c`, `x0,y0,z0`) and sub-seismic fault-rock props (`shear_zone_width`, `gouge_pctile`).
- **closure_parameters (per closure):** `fluid` (oil/gas/brine), `intersects_fault`, `intersects_onlap`, `intersects_salt`.

---

## 5. Properties extraction from 2D slices of the 3D sample

**Stage owner:** `scripts/images/images_generator.py` (`GraphImageExtractor`)

The DB gives *what exists*; this stage gives *what's visible*. It reads the properties graph to know which objects to look for, opens the build's zarr volumes, and turns 3D masks into 2D sections + segmentation masks. This is also where **mask-measured geometry** (fault dip, closure area) is later derived (§6.3).

### 5.1 Per-object 2D slicing
For each object type it loads the relevant zarr(s) and produces masks:

- **faults** — prefers the wrapper `faults/fault_XX.zarr` (one file per fault → clean per-fault masks, mapped back to the graph node via `original_fault_index`). Falls back to a merged `fault_segments` global mask if the wrapper files are missing (it explicitly refuses to fake per-fault splits from a combined binary mask). Fault masks are thinned to a **trace-like** line (skeletonize, or erosion XOR) so the target is the fault surface, not a fat zone.
- **closures** — split into connected components inside the fluid-specific zarr (`oil/gas/brine.zarr`), each component matched to a `closure_N` node; plus a merged all-closures mask.
- **salt** — connected components named by order.
- **onlap** — **AGGREGATE**: one union mask (`object_id = "onlap"`), *not* per-component. Onlap is a pervasive depositional surface pattern with no discrete objects, so connected-component splitting shattered it into dozens of unstable fragments ("built too much"); keeping only the aggregate makes it usable. Its measure is `area_pct` (coverage); its count `number_onlap_episodes` stays a 3D DB value.

For each object `_slice_by_mask` picks the **inline/crossline/timeslice index that maximizes mask area** (the slice where the object is most visible), keeps the seismic slice + the aligned binary mask, and retains the full 3D mask (`mask3d`) for scene compositing. Vertical views are transposed so depth reads downward.

### 5.2 The shared scene (`_build_scene`)
Per-object crops each on their own slice can't be composited into one picture. So `_build_scene` chooses, per view, a **single shared slice index** that maximizes *how many objects appear* (tie-broken by union area), renders **one grayscale seismic image** for that slice, and cuts **one mask per object registered to that same slice** (fit to the image shape → pixel-aligned). It also writes a multi-class **overlay preview** (red=fault, blue=closure, purple=salt…). Redundant per-type union masks are dropped when individual parts exist.

The result is `build_objects/images/{sample}/scene_position.json`: for each view, the shared `image_path`, `overlay_path`, and a list of objects each with `object_id`, `object_type`, `class_id`, `class_color`, `bbox`, `center`, and its own `mask_path`. **This file is the contract** between imaging and everything downstream — the 2D graph, QA, and dataset all read object geometry and masks from here.

Because image and every object mask come from the **same slice index, axis, and view**, they are spatially aligned; the CSV pairs `images[0]` with `masks[0]`.

---

## 6. Graph building

Two graphs are built per sample: a DB-grounded **properties graph**, then a **2D-position graph** that overlays scene geometry.

### 6.1 Properties graph
**Stage owner:** `scripts/graph/graph_generator.py` + `scripts/graph/graph_system.py`

`graph_generator.py` watches `success.yaml`, traces each build's DB **once** (idempotent: a lock + an on-disk graph check kill the O(N²) re-trace churn that `success.yaml`'s whole-file rewrites would otherwise cause), and calls `GraphSystem` with a **category filter**. The filter (`CATEGORY_FILTERS`) decides which tables and which keys per table are kept for each category — e.g. a `boring` sample never pulls `fault_parameters`.

`GraphSystem` builds a `networkx.MultiDiGraph`:

- a **category node** (`{category} structure`) carrying the model-level properties,
- a **type hub** per object class (`fault`, `closure`) linked `HAS_FAULT` / `HAS_CLOSURE`,
- **per-object nodes** (`fault_0`, `closure_1`, …) linked `has_object`.

Important transforms in `_add_by_filter`:

- **Visible-fault remap.** `fault_voxel_count_list` tells which faults actually rendered voxels. Invisible faults are dropped, `number_faults` is corrected to the visible count, and visible faults are **re-indexed densely** while stashing `original_fault_index` (so imaging can still find the right `fault_XX.zarr`).
- **Low-value pruning.** Properties equal to `False/0/None/[]` are removed (except `salt_inserted`), so the graph only asserts things that are true.
- **Display units (`_to_display_units`).** `throw` is converted to **milliseconds two-way time**: `throw / infill_factor × digi` — Synthoseis stores throw as fine-grid vertical samples, `÷ infill_factor` gives output-cube samples, and `× digi` (the ms sample rate; the cube's vertical axis is TWT) gives ms TWT, the conventional time-domain unit (metres would need a velocity model the sim doesn't provide). `tilt_pct` is ×100. (Note: `tilt_pct` is a rotation *fraction*, not a legible angle — since replaced downstream by mask-measured dip, §6.3; the conversion is left in place but no longer rendered.)

Serialized to `graphs/properties_graph/{sample}_..._properties_graph.json` as `{nodes:[…], edges:[…]}`.

### 6.2 2D-position graph
**Stage owner:** `scripts/graph/properties_2d_graph.py`

For each view (inline, crossline) it **deep-copies** the properties graph (the DB graph stays pristine) and overlays scene geometry from `scene_position.json`: it attaches `x`, `y`, `bbox` extents, `color`, and `view` to matching nodes, and **adds visual-only nodes** (with `HAS_VISUAL_OBJECT` edges) for objects the DB graph didn't already have. `lithology`/`age_depth` are excluded (`EXCLUDED_VISUAL_OBJECTS`); **onlap is kept as a single AGGREGATE object** (§5.1) — only stray numbered `onlap_N` parts are dropped. Output: `graphs/properties_2d_graph/{sample}_{view}_properties_2d_graph.json`, which also carries the shared `scene` block (image/overlay paths).

**View-filter (why the per-view graph is pruned).** A 3D fault need not intersect every 2D slice: the DB knows 5 faults, but an inline section may render only 4. The per-view graph files existed, but their *contents* were identical and unfiltered — so RAG could serve `fault_2`'s facts on a view where `fault_2` isn't in the picture, generating QA about an object with **no mask** (which DatasetMaker could only log as `[NEG-NAMED]` and emit as a bogus maskless "negative"). `_copy_graph_with_2d_positions` now **prunes object instances that have no position in this view** (`positions` is exactly the view's rendered objects) and drops edges touching them. Section/hub nodes (the category node, the type hubs) never carry a per-view position and are always kept.

**View-scoped recount — now for ALL types.** Pruning instances would otherwise leave the category node claiming the DB total ("7 faults" over a section showing 6), so every count is recomputed to what is **visible** in this section: `number_faults` → surviving fault instances, `number_hc_closures` → visible closures **whose fluid is oil/gas** (the HC subset — brine closures are water-bearing, not hydrocarbon, so they don't count), `salt_inserted` → false when no salt is visible in the view. `number_fault_intersections`, `fault_mode`, and `number_onlap_episodes` are *not* recomputed — they are 3D-structural counts with no per-slice instances (onlap is aggregate, §5.1). This closes the old "only faults were recounted" gap and lets closure/salt/onlap-heavy categories carry honest per-view counts.

### 6.3 Mask-measured visual attributes (dip, area)
All mask→attribute computation now lives in **`scripts/graph/compute_attribute.py`** (`mask_features`, `dip_degrees`, `area_pct`, `bbox_from_mask`, `center_from_bbox`, `centroid_from_mask`) — pulled out of `properties_2d_graph.py` and `images_generator.py` into one organised home so future attributes (perimeter, orientation, aspect, compactness, convexity, fault length/curvature — see the attribute table) have a place to land; `properties_2d_graph.py` imports `mask_features` and keeps only graph orchestration + the DB recount, and `images_generator._mask_bbox` delegates to `bbox_from_mask`. Computed **at 2D-graph time straight off each object's scene mask** — so the numbers reflect what's actually visible in pixels, not an opaque simulator knob:

- **fault `dip_deg`** — apparent dip = the angle of the fault trace's dominant line, 0°=flat … 90°=vertical, magnitude only. Estimation is **RANSAC + inlier gate**, not plain PCA: RANSAC finds the dominant collinear pixels and the angle is PCA-fit on those *inliers*, so a crossing structure or fragmented trace can't drag a moment-fit flat (which produced geologically absurd near-horizontal fault dips). A mask whose dominant line covers **< `_DIP_MIN_INLIER_FRAC` (0.5)** of the pixels returns `None`: on the now-individual per-object masks a low inlier fraction means the fault's *own* trace is non-planar (listric/branching) or fragmented — and a non-planar fault has no single dip (its pattern is still carried by the graph's `fault_mode`). On a clean single-fault mask every pixel is an inlier, so it reduces exactly to the old PCA angle. Deterministic (fixed RANSAC seed).
- **closure / salt / onlap `area_pct`** — mask coverage as a percentage of the section. (For onlap the aggregate mask *is* the object, so `_mask_features` measures it despite `object_id == object_type`.)

These are attached as node attributes, so they flow through the tracer → text transform automatically. This replaces the DB's opaque `tilt_pct` (dropped) and fills a gap the DB simply doesn't have (closure area). Only individual objects get them; the merged per-type mask (`object_id == type`) is skipped.

> **Apparent vs. true dip — a settled decision.** `dip_deg` is *apparent* dip measured from the view's mask, and that is the **correct** target for this dataset, not a compromise. Each row is one 2D section (one view); from a single slice only apparent dip is determinable — true structural dip needs the fault's strike (a second non-parallel section / 3D), so labeling a single-view input with true dip is *ill-posed* (the label isn't a function of the pixels). Apparent dip is view-dependent, which is exactly right: inline and crossline legitimately show different apparent dips, each matching its own image. Synthoseis stores no `dip`; a true 3D dip *is* computable from `tilt_pct` + fault origin + `cube_shape` (`θ = atan2(tilt_pct·√((x0−nx/2)²+(y0−ny/2)²), nz)`, `dip = 90°−θ`), but it is view-independent (wrong as a per-view label) and needs a full rebuild (origin/cube_shape are dropped from graphs). If a true-dip *attribute* is ever wanted, the cheap route is to combine the two views the pipeline already makes: `tan(dip_true) = √(tan²δ_inline + tan²δ_crossline)` — no re-slicing. The section aspect (~5:1, 507×100) makes apparent dip a lossy, many-to-one proxy for true dip (steep faults saturate near-vertical) — fine, because true dip shouldn't be recovered from one 2D slice anyway.

---

## 7. Graph → evidence → RAG

### 7.1 Tracing relations
**Stage owner:** `Tracer/tracer.py` (`EvidenceTracer`)

Walks the 2D graph and emits a flat list of **relations**: one per node property (`source, edge=key, target=value`) and one per edge (`source, HAS_*, target`), each with a `relation` triple and a raw `text`. This is deliberately exhaustive — it turns *every* surviving attribute into a candidate fact; the whitelist lives in the next stage.

### 7.2 Relations → natural evidence
**Stage owner:** `NaturalTransform/text_transform.py` (`TextTransform`)

Converts relations into short, inspectable English sentences — one sentence per triplet, no grammar variation. It's the **whitelist + renderer**:

- **`ALLOWED_PROPERTY_EDGES`** gates which facts become evidence: model counts (`number_faults`, `number_hc_closures`, `number_fault_intersections`, `number_onlap_episodes`), `salt_inserted`, `fault_mode`; fault `throw` + `dip_deg`; closure `fluid`, `intersects_*`, `area_pct`; onlap `area_pct`. (Sub-seismic `shear_zone_width`/`gouge_pctile` and the opaque `tilt_pct` are intentionally excluded.) `fault_mode` is further **whitelisted** to genuine geological patterns (`relay_ramp`, `horst_and_graben`, `branching`, `staircase`); `random` is a Synthoseis generation setting, not a structure, so it emits **no pattern sentence** rather than leaking the simulator into the evidence.
- **Objects are named by COORDINATE, not "Type N".** `node_name` renders each numbered object as `"the <type> at [x,y]"` using its center (from the position relations, via a per-view coord map) — e.g. `"the fault at [57.5,289]"` — for **every** object type. The internal id (`fault_2`) is unchanged (mask routing, `object_id`, linkage all still key on it); only the *displayed* reference changes. This lets the LM reason spatially ("the closure at [22,140] is left of the fault at [57.5,289]") because the position is in the text. Coordinates are plain `[x,y]` (no tags), and — since value tags were removed (below) — the separate `"sits near [x,y]"` position line and `"occupies the area from [x1,y1,x2,y2]"` extent line are plain too. (This replaced the old `Fault N` = internal_index+1 naming.)
- **Templates** render each as **plain prose, no value tags**: `"the fault at [57.5,289] has throw of about 124 ms"` (ms TWT, §6.1), `"… dips at about 58 degrees"`, `"the closure at [22,140] covers about 12 percent of the section"`, count/boolean/intersection phrasings, and grouped **position** (`"… sits near [x,y]"`) / **extent** (`"… occupies the area from [x1,y1,x2,y2]"`) sentences. The number and coordinate are the ground truth; nothing wraps them.
- **Derived readings (tier-2).** Beyond the raw facts, `TextTransform` emits deterministic, **source-backed geological readings as additional grounded evidence lines** sharing the object's `object_id` (so they mask the same object and route identically): `dip_deg` → `"appears steeply / moderately / gently dipping in this section"` (30/60° descriptive scale, scoped to *apparent* dip), `fluid` oil/gas → `"hydrocarbon-bearing closure"` / brine → `"water-bearing closure"`, `intersects_fault` → `"fault-dependent closure"` (grounded in Synthoseis's own closure-type logic, `Closures.py:1562`), `intersects_onlap` → `"onlap trap"`. Because they're real evidence, they are **retrievable and NLI-checkable exactly like the raw facts** — a question/answer may use the geological term and still clear the gates. Magnitude labels (large/small throw, big closure) are deliberately *not* derived: no sourced numeric cutoff exists, so they'd be arbitrary. This gives three grounding tiers: raw measured fact → deterministic derived reading → interpretive reasoning (§8, un-checked, prompt-bounded).
- **Value tags removed (2026-07-26) — evidence is plain prose.** Earlier, spans carrying entity/numeric identity were wrapped in `<object>`, `<nums>`, `<center>`, `<bbox>`. These are **no longer emitted**: `text_transform._tag` was neutralized in one place, so every evidence line is plain text with real values as bare tokens/coordinates (`… has throw of about 124 ms`; `the fault at [57.5,289]`; `sits near [x,y]`; `occupies the area from [x1,y1,x2,y2]`). The tags had become pure liability:
  1. they were **stripped anyway** before both embedding (§7.3) and NLI (§8.3), where their mere presence dropped same-fact cosine similarity from ~1.0 to **~0.62** and pushed real facts below `MIN_RETRIEVAL_SCORE` — removing them at the source eliminates that failure mode instead of patching it;
  2. they were **never needed for fidelity**, which is enforced by the prompt rule "keep every value/coordinate exactly as the Evidences give it" (§8.2) plus the coordinate entity guard (§8.3), not by tag-copying.

  **Only structural tags survive**, and they are added downstream by **DatasetMaker**, not `text_transform`: `<region>` / `<SEG>` inside `<evidence>` (§9). The model's **segmentation / value targets therefore come from those structural markers plus the per-region `values` in the `regions` column — not from inline value tags** (§9.4, §10). Coordinates stay in the plain text so the LM can still reason spatially.

Each evidence item keeps its structured fields (`object_id`, `edge`, `target`, `relation`, `trace_type`) so masks can be re-associated later.

### 7.3 RAG construction & retrieval
**Stage owner:** `Verifier/create_rag.py` (`Rag`)

- **Documents are OBJECT-LEVEL, not per-fact.** `prepare_content` groups every sentence by `object_id` into **one `Document` per object** (`page_content` = all its sentences joined), with the per-fact structure preserved in `metadata["facts"]` = `[{edge, target, text, trace_type}]`. Model-level facts group under the category node as a "section" pseudo-object. **Why:** per-fact documents fragmented one object across many docs, so a narrow query fanned out and pulled sibling objects' facts (pollution). With object-level docs, a query finds a *coherent object*, verification checks each answer fact against that object's whole fact set (§8.3), and masking stays one-object-one-region — while `metadata["facts"]` keeps evidence/mask routing per-fact.
- **Vector store:** `InMemoryVectorStore` with HuggingFace `all-MiniLM-L6-v2` embeddings, wrapped in **`_TagStrippingEmbeddings`**.
- **Tag-stripping embedder (now largely vestigial).** With value tags removed at the source (§7.2), the object-level documents are already plain, so there is nothing to strip on the evidence side; the `_TagStrippingEmbeddings` wrapper is kept only as a defensive no-op (it would still strip a stray `<SEG>`). *Historically this was essential:* embedding the old `<object>`/`<nums>`/`<center>`/`<bbox>` tags dropped same-fact similarity from ~1.0 to **~0.62**, so a tagged doc scored *below* `MIN_RETRIEVAL_SCORE` against the LLM's untagged `RETRIEVAL_QUERY` and got rejected — a large part of *why* the tags were ultimately dropped rather than merely stripped.
- **Graph retrieval:** `langchain_graph_retriever.GraphRetriever` with an `Eager` strategy (`start_k=6, k=20, select_k=20, max_depth=3`) traversing metadata edges (`object_id↔object_id`, `object_id↔parent_id`, `parent_id↔category_id`, `source↔parent_id`). The `edge↔edge` link was **deliberately removed** — it used to fan a narrow seed out to every object sharing a relation (all closures with `intersects_fault`), collapsing precision. **Breadth now comes from the query, not from cross-object fan-out.**

So retrieval isn't circular self-lookup: a query embeds, hits its nearest evidence docs, then walks a bounded neighborhood of the graph around them.

---

## 8. LLM, prompts, and QA generation

**Stage owners:** `Verifier/llm_machine.py` (prompts + model), `Verifier/generator_pipeline.py` (`RagWorkflow`, the loop)

### 8.1 The model
A **local sglang** server (`http://localhost:8000/v1`, launched by `scripts/run_sglang.sh`) running `Qwen/Qwen2.5-1.5B-Instruct` at **`--context-length 4096`** (raised from 2048: the richer new-schema evidence — a closure now carries ~9 lines — pushed the question prompt to a median ~1.5k / max ~3.3k tokens, and `input + max_tokens(640) > 2048` returned HTTP 400 on ~⅔ of graphs, a silent retry-storm that produced 0 rows; 4096 gives a 3.4k input budget). Driven via `langchain_openai.ChatOpenAI`. Two tuned client bindings: **question** (higher temp/penalties for variety) and **answer** (low temp for faithfulness). Outputs are Pydantic-parsed JSON (`QuestionBatchStructure`, `AnswerBatchStructure`) with a strict output contract and retries. (The `reason` client/parser/prompt were removed with the reason column.)

The shared `MASTER_PROMPT` sets a **senior seismic interpreter (structural geologist) persona** — a natural interpreter's voice, not a data readout — and enforces the truthfulness contract: only use the evidence; never invent objects/values/causes; never mention graph/metadata/synthetic/prompt/verification.

**Wording is now free; only values/coords are pinned.** The model may reword sentences and choose how to present a coordinate (`[x,y]`, `(x,y)`, "near x, y") — this is deliberately relaxed for variety and to let spatial comparisons read naturally. The one hard rule kept: *keep every numeric value and coordinate exactly as the Evidences give it (never invent, round, or change a number); always write a number as a **digit**, never a word ("2 faults", not "two faults"); and refer to each object by its coordinate as the Evidences do.* Answers no longer carry grounding tags at all (see §9.2 — the answer is emitted as plain text), so there is nothing for the model to tag; `RETRIEVAL_QUERY` stays verbatim (below) because verification looks it up. **The model also emits no `<SEG>` token** — segmentation markers are owned by DatasetMaker, injected into the `<evidence>` regions (one per mask, §9), not written by the LLM (which placed them mid-sentence and broke the format).

**`RETRIEVAL_QUERY` is a verbatim copy, not a paraphrase.** Both prompts require the exact Evidence line(s) a question/answer rests on, one per line. A paraphrased query embeds differently *and* mis-attributes at verification; copying is also easier for a small model than rewording. Compound (multi-fact, multi-object) questions and answers are explicitly allowed, because coverage verification (§8.3) checks each fact independently.

### 8.2 The QA loop (per 2D graph, per view)
The loop aims for `QUESTION_PER_GRAPH` (**5**) passing rows, capped at `MAX_ATTEMPT` (`3×` that = 15) outer attempts — right-sized to how few facts one 2D section actually carries, so it terminates early instead of grinding a fixed 200. Since a 6 h run is **throughput-limited** (~330–380 rows/hr on plain evidence, not corpus-limited), `QUESTION_PER_GRAPH` really controls the **spread**, not the total: at the measured rate Q=5 covers **all 408 graphs (204 scenes) in ~6 h** — full diversity, ~2 k rows, no idle time. Higher Q (12) drains rows off the *first* ~60 scenes before the window ends (concentration → overfit risk); lower Q (3) exhausts the corpus in ~3.5 h and under-uses the time. (Retuned across the run 12 → 8 → 5 as the value-tag removal ~doubled throughput; §13.1.) For each graph `RagWorkflow.generate_for_graph`:

1. **Seed evidence.** `evidence_seeds` yields, per seed, **`OBJECTS_PER_SEED` object docs (default 2) + the section doc** — so a batch shows the generator several coordinate-named objects at once (enabling multi-object / spatial-relational questions) *and* the section-level counts/mode. Set `OBJECTS_PER_SEED=1` for the old single-object behaviour; the section doc always rides along (the attaching-the-section-doc fix cured a bug where a lone-object seed made count answers collapse to "1 fault"). Two fixes here:
   - **Section split (🔴 bug fixed).** The object-vs-section split now keys on `_SECTION_ID_RE = " structure$"` (the `<category> structure` node holds counts/mode) instead of the old `_\d+$` "has a numeric suffix" test. That old test misfiled the **aggregate onlap** (`object_id "onlap"`, no `_N`) as a *section* doc — and section docs ride **every** seed, so onlap got asked about on every seed and ended up **~44% of all regions**. Now `onlap` is a normal object (one seed per scene), so its share drops toward parity **at the source** (no post-hoc balancing needed).
   - **Same-type pairing.** Objects are grouped by type (shuffled within/across groups) before chunking, so a 2-object seed is usually **same-type** (two faults / two closures) — which is what lets a **magnitude** comparison ("which dips more") exist; cross-type pairs only occur at a type boundary and fall back to a spatial relation.
   The `generate_for_graph` loop re-creates the generator on exhaustion, so each cycle re-shuffles into fresh groupings.
2. **Generate questions.** `question_batch_generation` produces natural GroundVQA-style questions (no tags, no leaked values), rotated across five **facets** — *presence/featureless, count, location, orientation/geometry, relationship between two named structures* (`QUESTION_FACETS`, shuffled per batch) — but *evidence-gated*: an angle the evidence can't support is skipped. Questions may be **simple or compound**. When the evidence describes **more than one object** (the usual case with `OBJECTS_PER_SEED=2`), the prompt asks the batch to cover **both**: a per-object **detail** question for *each* object, **and** at least one that **compares or relates two** of them — a *magnitude* comparison for same-type objects (two faults by dip/throw; enabled by same-type seed pairing, step 1) or a *spatial relation* for different types. This is answerable by construction — every object's full attributes are in the seed — so it is **not** the old *forced*-compound (which produced unanswerable clauses and an ~88%-rejection throughput collapse); it just stops a multi-object seed from yielding *only* single-object questions (which had left genuine 2-object comparisons at ~9%). Two guardrails still hold: **every clause must be answerable** (no subjective/visual-only clause like *"which is more prominent"* — no evidence edge), and a comparison must **name both objects by coordinate**. A comparative/superlative over a *single* object is disallowed (it is trivial). Each question ships a `RETRIEVAL_QUERY` = the **verbatim** Evidence line(s) it rests on.
3. **Retrieve for the question.** `retrieve_many(retrieval_query)` runs each query line through the graph retriever, dedups, and keeps only docs with `_similarity_score ≥ MIN_RETRIEVAL_SCORE (0.7)`. (0.7, not 0.9: MiniLM cosine rarely clears 0.9 for related-but-not-verbatim sentences; precision is still enforced by NLI coverage + coordinate entity guard + edge gate below, so retrieval only governs candidate **recall**. Tag-stripped embedding (§7.3) is what makes 0.7 reachable at all.) No docs → reject the question.
4. **Generate answers.** `answer_batch_generation` returns up to `CANDIDATE_PER_QUESTION` (**3**, trimmed from 5 for throughput) candidate answers, each with its own concise `RETRIEVAL_QUERY` (the claim, not the prose). The answer is **plain prose and emits NO `<SEG>` token** — segmentation targets are attached downstream by DatasetMaker from the `regions` (§9), so the model writing `<SEG>` only corrupts the answer format (it placed them mid-sentence, unbalanced); `_untag_answer` strips any stray one as a backstop. For a **comparative / superlative** question the answer prompt has the model read the evidence *like a geologist and make the call with confidence* — rank the values the evidence gives and name the winning object by coordinate, **stating both compared values** ("the fault at [57.5,289] is steeper, 65° vs 45°"). This is framed as a *grounded inference, not invention* — comparing/ranking given numbers is the interpreter's job — and reverse-NLI (§8.3) still checks the call back against the evidence, so a conclusion that follows from the numbers passes. The prompt also enforces **attribute discipline**: attribute a measured property (dip / throw / coverage %) to an object *only if the Evidences give it for that object* — onlap and salt have coverage + position but **no dip**, so the model must never call them "dipping"; for "which is steeply dipping" it must pick only among objects that carry a dip value (this is the generation-side half of the attribute-consistency guard, §8.3). Only the single best survives verification, so 3 (not 100) spreads phrasings without wasting generation/retrieval/NLI, and fits `max_tokens`.
5. **Ground + verify each answer** (`best_answer`) — see §8.3.
6. **Dedup by evidence.** The same evidence set may back at most `MAX_ROWS_PER_EVIDENCE (2)` rows, so identical images don't over-repeat. The key is now the **union** fact-set signature, so a throw+dip answer and a dip-only answer count as different grounding instead of colliding.
7. **Append the row** to `Dataset/verified_qa.jsonl` (dedup by `row_id`, atomic append + flush). Each row records question/answer/evidence, the verification score, `trace` (question/answer evidence), and `metadata` (graph path, view, scene image path, `graph_mtime`).

A `[TALLY]` per graph reports where attempts die (question reject / answer reject / row skip / passed) so you can see whether `MAX_ATTEMPT` is the bottleneck.

### 8.3 Verification: coverage, not a single entailment

The old design ran `filter_docs_by_trust` (one NLI check of the *whole answer* against each doc) plus a final `verify_answer`. That breaks on compound answers: a single-fact doc only partially matches a two-fact answer, hovers at the threshold, and facts silently lose their grounding. It is replaced by **`cover_answer`**:

- **Shared fact pool.** The pool is every fact of every retrieved object — the **question's docs *and* the answer's** (`object_docs = dedupe(question_docs + answer_docs)`). A fact the *question* retrieved can therefore ground the answer.
- **Coverage (forward pass).** The answer's `RETRIEVAL_QUERY` is already one fact per line, so it is split on `\n`; **every line must be entailed** by some pool fact at trust ≥ `MIN_ANSWER_TRUST` (**0.80**, see below), and is attributed to the best-entailing one. Any uncovered line → reject. (An earlier attempt to sentence-split the *answer* instead was reverted: it broke decimals — `80.5` → `80`,`5` — and stripped subjects, causing 14 false rejects.)
- **Forward vs. reverse (order matters).** Forward runs first *and is the gate*: if any declared `RETRIEVAL_QUERY` line is uncovered, `cover_answer` returns `None` and the reverse pass never runs. Reverse only runs on an already-passing answer and can only **add** facts, never reject — forward = *truth of declared claims*, reverse = *completeness of the compound*.
- **Compound completeness (reverse pass).** The small model often writes *one* query line for a two-fact answer, leaving the second fact real but ungrounded. So each pool fact of the object(s) the answer **cites by coordinate** is checked in reverse — `response=fact, sources=[answer]`, i.e. *"does the answer assert this fact?"* — and entailed facts are attached. Object identity is the coordinate (`coord_refs`), so only facts sharing the answer's cited coords enter the reverse pool (a different object's facts can't leak in). NLI decides; **no answer-splitting, no number-parsing**. Result: multi-evidence rows went **0 → 20/47**.
- **The NLI score itself** is a *hybrid* verifier (`longtracer`): bi-encoder semantic similarity (`avg_score`) gates a cross-encoder 3-class NLI (contradiction/neutral/entailment). A claim is supported when `avg_score ≥ 0.40` **or** `entailment > 0.5`, **and not** `contradiction > 0.5`; the thresholded `trust_score` is the similarity, the NLI drives the PASS/contradiction gate. The pipeline requires `PASS ∧ trust ≥ MIN_ANSWER_TRUST`, **raised 0.7 → 0.80 for the answer gate** (retrieval recall stays 0.7; question grounding stays lenient at 0.7). Rationale, from the actual score distribution: good answers ground verbatim-ish and cluster near **0.98 (median)**, so a 0.80 floor trims only the thin **low-confidence tail** (~1% of good rows) where loosely-grounded junk lives — e.g. the attribute swap that scored **0.79** on shared-subject similarity. It is a **blunt confidence floor, not a semantic fix** — STS similarity ≠ entailment, so a *same-object* attribute swap can still score 0.9+ — hence it is *complementary* to the structural guards below, not a replacement; pushing past ~0.82 costs real yield (~15% of good rows gone at 0.85). The contradiction guard is what rejects a wrong-*valued* fact ("dips 40°" when the answer says 65.8°).
- **NLI label-order bug — found & fixed (`Verifier/nli_patch.py`).** longtracer read the `cross-encoder/nli-deberta-v3-*` logits — whose real order (per the model's own `id2label`) is `[contradiction, entailment, neutral]` — as `[contradiction, neutral, entailment]`, **swapping entailment↔neutral**. `contradiction` (index 0) was correct, so value-errors were still caught, but every `entailment`/`neutral` read was **inverted**: a *neutral* claim ("onlap is dipping" grounded on an area fact) reported as **0.99 entailment** and passed, while a true paraphrase reported as **0.01**. A monkeypatch re-indexes to the model's real order, applied at `generator_pipeline` import (`apply_nli_label_fix`). Verified: the swap now reads `entail 0.001`, true paraphrases `~0.99`. The model itself (`xsmall`) is adequate — **no upgrade needed; it was a library bug, not a weak model.**
- **Entailment requirement (Fix 2).** With labels corrected, `cover_answer` now requires a claim line's `entailment > MIN_ENTAILMENT (0.5)` — not similarity alone — to count as covered (`_entailment_of`, on both forward and reverse passes). This is what the `avg_score ≥ 0.40 OR entailment` clause was *meant* to enforce; before the label fix that clause read the *neutral* probability, so neutral claims slipped through. Verified not to hurt yield (true dip/throw/count and reworded paraphrases all still cover).
- **Plain text end-to-end (was: tag-stripped NLI).** Evidence and answers are now plain prose (§7.2), so NLI compares clean text directly. *Historically* NLI ran on tag-stripped text because tags depressed scores — a true throw fact scored **0.616 tagged vs 0.823 stripped** — and that gap is exactly why value tags were removed at the source. The strip step is retained defensively (harmless when there is nothing to strip, e.g. a stray `<SEG>`).
- **Entity-swap guard** (`answer_objects_in_docs`) — now **coordinate-based**. Since objects are named by coordinate, every coordinate group the answer cites (center `[x,y]` or box `[x1,y1,x2,y2]`, any bracket style, number-normalized, matched as *whole groups* so a box is never mistaken for two centers) must appear in the evidence. A coord not in the evidence is a swap/fabrication → reject; NLI still judges the *relation* on top ("A at [..] is left of B"). (This replaced the old `<object>`-string + "Type N" match, which coord-naming and free rewording made unworkable.)
- **Question-coverage edge gate** — the answer must address what the question *asks*, not merely be grounded. This replaced the old keyword *facet* gate: both the question and the answer are grounded to per-fact evidence carrying `metadata["edge"]`, so the gate compares **edge sets** directly — the answer's grounded edges must **cover** the question's grounded edges. A compound question grounds to ≥2 edges (e.g. `dip_deg` + `number_faults`), so an answer that drops a clause is missing that edge and is rejected — precisely the failure the keyword any-overlap facet gate let through. When the question grounds to nothing, it passes (lenient, no false-reject).
- **Topic + count guards** (`object_types_in`, `count_edge_for_question`, applied right after the edge gate) close two cross-wiring failures the edge gate misses — because a **count question often grounds to no edge**, leaving `q_edges` empty so the edge gate is skipped, and any grounded fact then slips through. **Topic guard:** if the object type(s) the *question* names (`fault`/`closure`/`salt`/`onlap`) and the *answer's* object type(s) are **disjoint**, reject — this kills "how many faults?" → "salt is present". **Count guard:** a "how many X" question must ground the matching `number_*` edge in its answer (`number_faults`/`number_hc_closures`/`number_onlap_episodes`/`number_fault_intersections`), else reject. Both are conservative (only a *clearly* off-topic answer is dropped, so compound/comparative answers pass). Together they took cross-type mismatches from **~25% → 0%** on the multi-type scenes.
- **Attribute-consistency guard** (`attr_edges_required`, right after the count guard) — a **magnitude** the answer's wording asserts must be grounded on the matching edge: a *dip* claim ("dips", "N degrees") requires a `dip_deg` edge, a *throw* claim requires `throw`, a *coverage-%* claim ("covers N percent") requires `area_pct`. Otherwise the answer claims a property the object has no evidence for — *"onlap appears steeply dipping"* grounded only on `area_pct`, or *"onlap covers 20%"* grounded only on `extent` (the 20% ungrounded/leaked). Patterns require a real value/keyword, **not** the bare word "area" — so *"occupies the area from [bbox]"* (an **extent** statement, legitimately grounded on `extent`) does not trip it, nor does *"larger in area"* inferred by comparing two bboxes. This closes a hole the NLI structurally cannot: the swap scores high on *shared-subject* similarity (§ the 0.80 floor above catches the loose ones, but a same-object swap can score 0.9+). It is the **deterministic half of a two-layer fix** — the other half is an **answer-prompt rule** (§8.2 step 4): the model is told to attribute dip/throw/coverage to an object *only if the Evidences give that property for that object* ("onlap and salt have coverage + position but no dip, so never call them dipping"), which cuts these at **generation** and saves the generate-then-reject throughput. Measured **hard** mismatch rate before the fix: **~12%** (dip pinned on non-dipping objects; ungrounded percentages); the ~303 clean rows already produced were retro-filtered against both this guard and the 0.80 floor.
- **Why the structural guards can't be replaced by NLI (tested, not assumed).** Once entailment is trustworthy (label fix above) it is tempting to delete the topic/count/attribute guards. Two tests show NLI **structurally cannot** do their job. (1) `cover_answer` verifies the model's `RETRIEVAL_QUERY` — a *true* fact — **not the answer's wording**; so a benign query ("onlap covers 1.1%") masks a mismatched answer ("onlap is dipping"), which **only the attribute guard** catches. Confirmed: that swap stays COVERED through `cover_answer`, rejected only by the guard. (2) A whole-answer *"evidence ⊨ answer"* check is a **dead end** — it false-rejects **grounded inferences**: "A is steeper than B, 65 vs 45" scores `entailment 0.004 / contradiction 0.505` against its own two dip facts, because the NLI can't see that *steeper* follows from *65 > 45*. The **edge-based guards** separate grounded inference (dip claim **with** dip edges → pass) from hallucination (dip claim, **no** dip edge → reject) exactly where NLI conflates them. So the guards are the **correct tool**, not a workaround — the earlier "broken source" was the NLI label bug (now fixed), not the guards.
- **Verdict:** `{"verdict": "PASS", "score": mean(covered trust)}`. Best-scoring surviving answer wins.

The **question** is also grounded per-fact (`cover_answer(..., require_all=False)`, lenient — question evidence is additive, not a gate), and the row's evidence is the **union of question ∪ answer facts**. This both completes the grounding (a comparison question keeps the objects it compares) and *repairs* some answer-side mislabels: for "how many fault intersections?", the answer sometimes grounds `number_faults` while the **question** correctly grounds `number_fault_intersections` — the union carries the right fact.

> **Why this design:** questions and answers come from the *same* small model but must survive **retrieval gating + NLI coverage + coordinate entity guard + edge gate** against graph-derived evidence. The graph is the source of truth; the LLM only phrases. Note the honest boundary: verification checks the answer against **the facts its own `RETRIEVAL_QUERY` fetched**, so a *self-consistent* wrong answer can pass (§12).

### 8.4 Running QA at scale: clean-regen, NLI-on-CPU, sharded workers, resume

**Entry point.** `scripts/qa_new_only.py` calls `generate_multimodal_dataset(graph_root, output_path, resume)` — a thin **standalone driver**, not the watcher. It pins `longtracer`'s NLI/STS verifier to **CPU** before the first `check()` (mirrors the watcher's `_init_nli_device`; the standalone path otherwise lets DeBERTa grab CUDA and OOM against the sglang server on the shared 6 GB GPU — `CUBLAS_STATUS_ALLOC_FAILED`). It also runs with `CUDA_VISIBLE_DEVICES=` so nothing in the QA process touches the GPU; the LLM is reached only over HTTP.

**Scope by `graph_root`.** `graph_root` is a parameter, so QA is scoped by pointing it at a symlink directory. The 2026-07-26 run QA'd **only the 204 new diverse scenes** (`graphs/_new_only/`, 408 graphs × 2 views) — the old fault_complex graphs were built with the pre-tonight extraction, so excluding them is right on both **balance** and **schema** grounds.

**Clean regen vs. resume.**
- **Clean** (`truncate=True`, the default): wipes `verified_qa.jsonl` once at start, then loops all graphs. Use after any prompt/schema change — the pre-existing rows were old-schema (plain answers, no coordinate naming, no `<SEG>`), so appending would mix schemas. The stale file is backed up first (`verified_qa.PRE_REGEN_0726.jsonl`).
- **Resume** (`resume=True` via `QA_RESUME=1`): `start_output(truncate=False)` appends and re-seeds `_seen_row_ids` for row dedup, and `_processed_graphs(output)` returns every `metadata.graph_path` already written so the loop **skips done graphs**. A run can be stopped and continued later with **no work redone** — the way a multi-thousand-row target is reached across several sessions.

**Sharded parallelism.** The pipeline is serial and **NLI-on-CPU-bound**, while the sglang server sits ~99% idle — so throughput scales by running **N workers over disjoint graph shards** (`graphs/_shard_0..4`, ~82 graphs each), each writing its own `verified_qa_shard_i.jsonl`. On the **12-core / 15 GB** box, **5 workers** is the sweet spot: each is ~1.7 GB RAM (→ ~6 max before RAM is gone) and pinned to `OMP_NUM_THREADS=2` so they don't oversubscribe the cores; sglang batches their concurrent requests. Result: **~30 rows/hr (1 worker) → ~330–380 rows/hr (5 workers, plain evidence)**, an ~11–12× lift (tagged evidence ran ~157 rows/hr — value-tag removal was a 2.2× throughput win, §13.1). NLI cannot move to GPU (sglang fills the 6 GB card), so CPU cores are the hard throughput wall.

**Ops.** `scripts/qa_shards.sh {start|resume|stop|status}` manages the workers (systemd `--user` transient units `seismic-qa-0..4`, memory-capped); `scripts/qa_shard_monitor.sh` logs combined rows / worker liveness / RAM and wakes on a dead worker or RAM-critical. At the end the shards are **merged** into `Dataset/verified_qa.jsonl`, then DatasetMaker builds the CSV.

---

## 9. Dataset making

**Stage owner:** `Dataset/DatasetMaker.py`

Reads `verified_qa.jsonl` and emits `Dataset/multimodal_multi_image_dataset.csv` (and a jsonl).

### 9.1 Regions and the row mask

1. Load the shared scene image + its objects from `scene_position.json`.
2. **Split section-scoped from object-scoped evidence.** `SECTION_EDGES` (= the `EDGE_TYPES` keys: `number_faults`, `fault_mode`, `number_fault_intersections`, `salt_inserted`, `number_hc_closures`) describe the **whole section**, not one object. Only object-scoped facts drive the per-object loop.
3. **Match evidence to regions** (`evidence_matches_region`): an evidence item lights up an object region when its `object_id` matches, or its type matches a type-global region, or it's an edge-type fact for that class. The subtle rule: object-specific evidence like `fault_0` falls back to the **type-global** mask *only when that object has no individual mask* (`individual_ids` guard) — so "dip of fault 1" masks fault 1 if a per-fault mask exists, else the all-faults mask, but **never** bleeds onto `fault_1` from `fault_0` evidence.
4. **Section facts → ONE whole-section region.** Previously a section fact matched *every* fault, so the per-object loop stamped an **identical** `<region>` block onto each one — 5 byte-identical blocks, all pointing at the same `mask_idx=0`, i.e. *one mask described five times*, teaching the VLM that each individual fault "is 17 intersections". Now section facts are emitted **once** as a single region (`object_id="section"`, union bbox over the referenced type, mask still covering all of them), so the (text → segment) pairing is 1:1 like every object row.
5. **One mask per region** (`build_region_mask`), **not** a single composite. Each region gets its own binary PNG under `Dataset/masks/` (keyed `_r{idx}_`): an object region = that one object's mask; the whole-section region = the **union** of every object of the referenced type. A region is only emitted if its mask has real pixels. So one row is **1 image → N masks → N evidence blocks → N `regions`**, all index-aligned: `masks[i] ↔ regions[i] ↔ the i-th `<region>`/`<SEG>``. Each region's `mask_idx` points at its own mask. (Verified against real data: region `bbox`/`center` reproduce the mask pixels exactly, and the evidence's plain position/extent coordinates equal the region's.)
6. **No-evidence rows fall back to the DEFAULT SHARED SCENE.** A featureless / section-only row (no object region built) no longer goes maskless. It's emitted with the scene image + a **whole-scene mask** — the union of every object, or a full-frame mask if there's nothing to outline (`_default_scene_mask`) — as **one global region** with its own `<SEG>`. So *every* row carries image + mask + region + `<SEG>` (the empty-mask negative is gone: 0 maskless rows, was ~142). The global region's `object_name` is `"the section"` (whole-section reference); its seismic class lives in `object_type`/`class_id`, derived from the section fact's edge (e.g. `number_faults → fault`).

### 9.2 The answer is plain; the evidence carries the ground truth

**Both the answer and the evidence text are plain** — value tags were removed at the source (§7.2), and `_untag_answer` strips any stray markup (incl. a `<SEG>`) from the answer as a backstop. Answer: `<answer>…</answer>` around plain prose. Evidence: `<evidence>` wrapping one `<region>\n{plain facts}\n<SEG>\n</region>` block **per mask**, e.g. `the fault at [57.5,289] dips at about 65.8 degrees.` — plain prose, real value, the coordinate names the object, no `<nums>`/`<object>` tag. Only the **structural** `<region>`/`<SEG>` tags remain, and DatasetMaker (not the LLM) injects them. Invariant checked at build across all rows: `#<region> == #<SEG> == #</region> == len(masks) == len(regions)`, and each `<region>` holds exactly one `<SEG>`.

**The structured grounding moved to the `regions` column** (the tags' *job*, not the tags). Each `regions[i]` (one per `<SEG>`/mask) carries:
- **`object_name`** — the object's coordinate reference (`"the fault at [57.5,289]"`) for an individual object, or `"the section"` for a global/section region. This *replaces* `object_id` — the internal id is never surfaced. The seismic class is carried separately by `object_type`/`class_id`.
- **`seg_idx`** — ties this region to the i-th `<SEG>`.
- **`values`** — the machine-readable ground truth, split by **value type** (which is also the head split): `{"measure": {…continuous scalar magnitudes: dip_deg, throw, area_pct…}, "derive": {…counts / categories / booleans: number_*, fault_mode, fluid, salt_inserted, intersects_*…}}`. `measure` are **regression** targets, `derive` are **classification/count** targets. The rule is the value's type, not where it was fetched — so `throw` sits with `dip` in `measure` even though it comes from the DB. Per type: **fault** measure `dip_deg`,`throw` / derive `number_faults`,`fault_mode`,`number_fault_intersections`; **closure** measure `area_pct` / derive `fluid`,`intersects_*`,`number_hc_closures`; **salt** measure `area_pct` / derive `salt_inserted`; **onlap** measure `area_pct` / derive `number_onlap_episodes`,`intersects_onlap`. `position`/`extent` are **not** in `values` (they are coordinates, already in `bbox`/`center`). Both keys always present; the same values also read as words in the evidence text (retrievable).
- plus `class_id`, `bbox`, `center`, `mask_idx`, `object_type`, `view`.

> **Why values live in `regions`, not inline tags.** Unwrapping the evidence keeps it retrievable/readable while the *exact* regression targets sit in `regions[i].values` (deterministic, no parsing of free text). Slots/placeholders in the text were tried and reverted — pre-blanking destroys the ground truth. The dataset is the regression *target*; the model's training preprocessing maps a `<SEG>`/value to its `regions` entry.

Columns: `sample_id, images, masks, instruction, question, answer, evidence, regions`. (The former `question_regions` / `answer_regions` columns and the `region_grounded_variant` twin maker were **removed**.) `Dataset/upload_to_huggingface.py` pushes it to the Hub.

### 9.4 The grounding format & the model-side contract

The format follows **GLaMM/LISA**-style *inline* grounding (not GranD's sidecar). Where grounding lives, in the current schema:

- **Segmentation** → the `masks` + `regions` columns. `regions[i]` carries `{object_name, class_id, bbox, center, values, seg_idx, mask_idx, object_type, view}` for `masks[i]`, one per `<SEG>`. **Referring-segmentation is prompt-driven** (§8.2): the question prompt favors casual, spatial localization ("segment the fault at [x,y]", "outline the largest closure"), and the answer names each localized object **by coordinate**. Crucially the answer is **plain prose that states the grounded facts and emits NO `<SEG>`** — it is verified by the *same* pipeline (retrieval + NLI coverage + edge/topic/count gates + coord swap guard). **DatasetMaker owns `<SEG>`**: it injects one `<SEG>` per region into the `<evidence>` (`<region>…<SEG>…</region>`, one per mask), where `seg_idx` is the `<SEG>`↔region link the collator uses. This replaced an earlier design where the 1.5B wrote `<SEG>` into the answer itself — it placed them mid-sentence and unbalanced, corrupting the format — so `<SEG>` was moved off the model and made deterministic; `_untag_answer` strips any stray token from the answer. (The build-time `[SEG MISMATCH]` log — answer `<SEG>` count vs region count — is now vestigial, since the answer carries none.)
- **Segmentation / value targets** → the **`<SEG>` markers + the `regions` column**, no longer inline value tags. DatasetMaker injects one `<region>…<SEG>…</region>` block per mask into the `evidence`, and each `regions[i]` carries that object's `values` (`measure` = mask-computed magnitudes dip/throw/area; `derive` = DB counts/categories, §9.1), plus `bbox`, `center`, `class_id`, aligned to `masks[i]`. Training preprocessing substitutes each `<SEG>` span with a special token whose target is the region's **mask** (GLaMM/LISA-style); the scalar ground-truth values ride in the plain evidence text *and* structured in `regions.values`. (These values were previously wrapped in `<nums>`/`<center>`/`<bbox>` spans; the tags were removed 2026-07-26 — the values are unchanged, only the markup is gone, so the collator still never asks the model to emit coordinate digits as free text.)
- **Object reference** → a **plain coordinate in the text** (`the fault at [57.5,289]`). This is a spatial *reference* for reading/generation and comparison, **not** a regressed token — the current model does not consume a `<bbox>`-token RoIAlign input, so object references are deliberately plain (no tag).
- **Answer** → plain natural language (no tags); it states values/coords faithfully (§8.2) and its object identities are pinned to real evidence coordinates by the swap guard (§8.3).

Contract notes:

- **The dataset is the annotation; the model target is derived** from the tagged evidence spans, not by parsing free-text digits.
- **Alignment is index-positional** end-to-end: `masks[i] ↔ regions[i] ↔ the i-th `<region>`/`<SEG>`` in evidence (invariant checked at build, §9.2).
- **Inference UX this buys:** the user can ask **by coordinate** ("what is the fault at [x,y]?") or by plain description — no need to know fault names — because the position is the reference, and it's the same form the training data uses.
- **Fault ids stay Synthoseis's internal enumeration** (`fault_2`) as the *internal* handle for mask routing and `object_id`. The numbering reflects **insertion/construction order** (structurally real), *not* spatial order — which is exactly why the **displayed** reference is now the coordinate, not a "Fault N" name. The old `Fault N` = internal_index + 1 display name was removed; the id is never surfaced to the model.
- **Possible extension** (not current): promote the question's object reference from a plain coordinate to a `<bbox>` RoIAlign-token input, restoring the box→features path — the `regions` metadata already carries the box needed to feed it.

---

## 10. End-to-end data flow

```
settings.yaml + control_parameter.py
        │  populate()
        ▼
recipes/recipe_N.yaml  +  build_configs/*.json
        │  build_sample → guarded_build_model (SUBPROCESS-ISOLATED, per BUILD_CONCURRENCY)
        ▼
builds/seismic__..._{category}_{uuid}/   (parameters.db + zarr cubes + faults/fault_XX.zarr)
        │  success.yaml
        ├─► ParameterDbTracer  →  graphs/{sample}_db_extract.json      (raw DB tables)
        │        │  GraphSystem + CATEGORY_FILTERS
        │        ▼
        │   graphs/properties_graph/{sample}_..._properties_graph.json (DB-grounded graph)
        │        │  GraphImageExtractor
        │        ▼
        │   build_objects/images/{sample}/scene_position.json + PNGs   (shared scene, per-object masks)
        │        │  properties_2d_graph.main  (+ dip/area from masks)
        │        ▼
        │   graphs/properties_2d_graph/{sample}_{view}_..._2d_graph.json   (VIEW-FILTERED + recounted)
        │        │  EvidenceTracer → TextTransform → Rag (object-level docs, tag-stripped embed + graph retrieval)
        │        │  RagWorkflow: question → retrieve → answer → cover_answer (coverage + reverse-NLI,
        │        │               question∪answer union) → coord entity guard → edge gate
        │        ▼
        │   Dataset/verified_qa.jsonl
        │        │  DatasetMaker: evidence→regions (section facts → ONE region), ONE mask per region,
        │        │               plain answer + <evidence>/<region>/<SEG> (ground-truth values), keep negatives
        │        ▼
        └─► Dataset/multimodal_multi_image_dataset.csv  +  Dataset/masks/*.png
```

---

## 11. Orchestration (the watchers)

**Stage owner:** `scripts/watcher/process.py`

Six `asyncio` coroutines run concurrently via `watchfiles.awatch`, each doing a **catch-up pass** over existing files then watching for new ones, so a fresh finished sample flows through immediately (not after the whole batch):

1. `recipes_watcher` — recipe add/modify/delete → maintain build configs.
2. `builds_config_watcher` — new build config → launch a build as an isolated subprocess (`asyncio.create_subprocess_exec` → the `sample_generator` CLI).
3. `builds_watcher` — `success.yaml` → trace DB → properties graph; `failed.yaml` → clean up. Also reconciles orphaned builds on startup.
4. `graph_properties_watcher` — new properties graph → `_run_image_gen` (images → 2D graph → delete the heavy build folder).
5. `dataset_gen_pipeline` — new 2D graph → enqueue.
6. `dataset_worker` — single consumer: coalesce the queue, generate rows per new inline/crossline graph, rebuild the CSV.

**Concurrency rules that matter:**

- **Builds run in isolated subprocesses**, not watcher threads. Synthoseis monkeypatches process-global state and is CPU-bound, so running it in-process would both leak patches into the watcher and hold the event loop's GIL (starving image/graph/QA stages while a build churns). A subprocess (awaited via `asyncio.create_subprocess_exec`) fixes both. Concurrency is `BUILD_CONCURRENCY` (default **1** == the old serial behavior, now off-GIL); `>1` runs builds in parallel and is safe because the success/failed trackers are guarded by a cross-process `fcntl.flock`. Everything else downstream still runs via `asyncio.to_thread`, image gen bounded by `IMAGE_GEN_CONCURRENCY`.
- **NLI on CPU** (`_init_nli_on_cpu`) so longtracer's verifier doesn't fight the LLM for VRAM; `_cuda_cleanup` after OOMs.
- **Idempotent / resumable.** Graphs traced once (lock + on-disk check); the dataset worker seeds "already processed" graphs by comparing stored `graph_mtime` (a rebuilt graph is reprocessed). `DATASET_REGEN=1` forces a full rebuild after a prompt change.
- **Clean Ctrl+C.** Low-level `signal.signal` handlers cancel the main task, and the process `os._exit`s rather than blocking on Python 3.11's `shutdown_default_executor()` waiting on an in-flight build thread. Partial builds (no `parameters.db`) are swept on the next start (`_sweep_partial_builds`).

### 11.1 Populating at scale (`scripts/run_generation.sh`)

**A build only starts if its config is `Change.added` *while the watcher is running*.** `builds_config_watcher`'s startup loop only *skips* already-built configs — it never builds pre-existing ones (`# what failed is failed, no rebuild from config`). So the order is non-negotiable: **start the watcher first, then populate.** Populating first and starting the watcher after silently builds nothing.

`run_generation.sh` is the detached driver for long runs:

- `setsid`-launches the watcher (survives shell/session exit) and **restarts it if it dies**,
- keeps a **shallow queue** (`QUEUE_AHEAD`, **3** in the balanced run) by calling `control_parameter` repeatedly — each call now emits a **weighted-random mix across all 8 sample types** (`control_parameter.populate()` draws each recipe from `ratio_per_types` via `random.choices`, converging to the target balance over many small recipes) instead of the old fixed *2 `fault_complex`*,
- stops at `TARGET_SCENES` or if free disk falls below `MIN_FREE_GB`,
- logs progress to `gen_driver.log`, build detail to `gen_run.log`.

**Measured cost (SKIP_QA build run):** ~**2 min/scene** (build + image gen, no inline QA) and ~**20 MB/scene** retained (`build_objects` = 8.9 GB / 446 scenes). Disk is *not* the long-term constraint; **wall-clock is**, plus **transient raw-build peaks** (the heavy zarr build folder before `_run_image_gen` deletes it) — those peaks, not retained output, are what pressure free space during a run.

> **Restart caveat:** a watcher restart re-QAs already-built graphs and **appends**, over-sampling those scenes. During a long build the QA is therefore provisional — finish with **one clean QA pass** (reset `verified_qa.jsonl`, regenerate over all graphs) so every scene is sampled exactly once.

> **Orphan-stall (long-run failure mode, observed).** Because a build only fires on `Change.added` *while watching* (§11.1) and the startup loop never rebuilds pre-existing configs, **any unit/watcher recycle orphans the configs that were in flight**: their add-events are already consumed, so they sit unbuilt, and once `pending ≥ QUEUE_AHEAD` the driver stops populating — a silent deadlock (no crash, no log, scenes frozen). Mitigated **externally, without touching `process.py`**, by `scripts/gen_monitor.sh` (the overnight watchdog): when it sees `pending ≥ 3` with **no build running** (`raw==0`) for two consecutive checks, it moves the settled unbuilt configs to `build_configs_orphaned/`, which drops `pending`, re-triggers the driver, and logs `STALL_HEAL`. It fired ~6× over a 7 h run and self-recovered each time.

> **Memory containment.** On a 15 GB box with training resident, `systemd-oomd` pressure-kills the whole unit during a multi-GB build peak → recycle → orphan-stall. Fixed by running the unit under a cgroup cap (`MemoryMax=8G`, `MemoryHigh=6500M`): builds throttle instead of triggering a system-wide OOM kill. With the cap, 0 unit-crashing OOMs across the run. NLI stays on CPU (`NLI_DEVICE=cpu`) so it never competes with training for VRAM.

---

## 12. Design rationale & known limitations

**Why grounded-and-verified instead of just prompting a big model:** the entire value proposition is *trust*. A seismic VLM trained on plausible-but-wrong answers learns nothing useful. Every number traces to `parameters.db` or a measured pixel region; NLI entailment + the entity guard reject drift; tagged spans force verbatim fidelity.

**Why measure geometry from masks (dip/area) instead of the DB:** the DB's `tilt_pct` is a simulator rotation *fraction*, not a legible angle, and it stores no closure area at all. A VLM can only verify what's in the pixels, so the visible geometry is measured from the same masks the model will see. The DB stays the oracle for *identity, counts, fluids, intersections*; masks own *visible geometry*.

**Honest limitations (also tracked in `README.md` TODOs):**

- **Single simulator ceiling.** Synthoseis defines the visual domain; expect a domain gap to field seismic. This is pretraining/augmentation data, not a replacement for real-data fine-tuning.
- **Text-mediated verification of a visual task.** QA is generated and NLI-checked against *text* evidence derived from ground truth; it asserts the answer is *true of the scene*, not that it's *visually inferable* from that particular 2D section. Mask-measured dip/area narrow this gap; it doesn't fully close it.
- **Verification follows the model's own query, so a self-consistent wrong answer can pass** — *largely mitigated now by the edge gate (§8.3), but not fully closed.* `RETRIEVAL_QUERY` is written by the 1.5B; retrieval faithfully returns whatever it asks for. The old failure: on a count question the model wrote a query about *one object* and answered "there is one fault present," grounded on that object's dip fact — self-consistent, passed at trust 1.0 even when the evidence said `number_faults=7`. The **edge gate** now catches this: the question grounds the `number_faults` edge, so an answer whose grounded edges don't *cover* `number_faults` is rejected — a count answer can no longer sail through on a dip fact. What remains: if the answer grounds the *right* edge with a self-consistently wrong value that NLI's contradiction check doesn't catch, it can still pass. Remaining lever: a **bigger LLM**, or a narrow **count guard** (for `number_*` questions require the answer's integer to equal the count edge's target — hook ready, unimplemented).
- **Small generator model.** Qwen2.5-1.5B is fast and cheap but a weak writer: it emits **0 tags**, under-lists `RETRIEVAL_QUERY` on compound answers (hence the reverse-NLI pass), and confuses presence with count. The gates catch most of it but also throttle yield (watch the `[TALLY]`).
- **Per-fault masks depend on the wrapper.** If `faults/fault_XX.zarr` isn't emitted (guard off / older builds), faults fall back to a single merged mask and lose per-fault dip/localization.
- **Combinatorial masks ≠ visual diversity.** Subset-masks from one complex scene are correlated; effective visual N ≈ **number of distinct builds**. Split train/val **by build** to avoid leakage. The QA:image ratio is *not* the overfitting driver (the ~12 QA/image land on **different facts**, deduped by evidence); **scene count is**. For a fixed row budget prefer **more scenes × fewer QA** over the reverse.
- **All four types now enabled (fault, closure, salt, onlap), recount extended to all — now proven by the balanced rebuild.** Onlap is aggregate (one mask), the per-view recount fixes `number_faults`, `number_hc_closures` (HC subset) and `salt_inserted`, and `area_pct` covers closure/salt/onlap. Previously staged code that had only run for `fault_complex`; the **overnight balanced run (2026-07-26) built 204 new scenes across all 8 categories** — `salt_fault_mixed` 32, `fault_only` 30, `depositional` 26, `onlap` 22, `salt_only` 18, `boring` 15, `full_mixed` 14 (+ `fault_complex`) — so the closure/salt/onlap extraction + recount paths all executed without error. What is *not* yet verified is the **numeric correctness of the new-type attributes**, which is a QA-pass check (pending, §13). `lithology` stays excluded by design.
- **`number_onlap_episodes` is a 3D DB count, not per-view recounted** — onlap is aggregate, so there are no per-surface instances to count per section; if onlap isn't visible in a view the count can slightly overstate. The one count that isn't visible-recounted.

---

## 13. Scale & the current run

**Corpus after the 2026-07-26 balanced run: 446 scenes** (~908 2D graphs, ×2 views ≈ **892 distinct images**) — scenes, not rows, are the unit that matters (§12: effective visual N ≈ distinct builds). The run targeted 450 and was **stopped 4 short at 446** to hold disk headroom for QA (free space had fallen to the ~18–20 GB floor). Composition:

- **242 older scenes** — the pre-balance corpus, essentially all `fault_complex`.
- **204 new scenes** — the balanced mix across all 8 types (breakdown in §12).

The **full 446-scene corpus therefore skews `fault_complex` (289 / 446 ≈ 65%)** because it carries the old fault-only scenes; the **new 204 alone are balanced**. Which of these to QA is the open **scope decision** below.

**Why 12 QA/image is the "optimal" pick:** it sits in the healthy VQA band (5–15 Q/image, cf. GQA/VQAv2) and well inside a scene's fact budget (~5–7 faults × ~5 facts + section facts ≈ 30+), so the 12 questions land on **distinct facts** rather than rephrasing one — and `MAX_ROWS_PER_EVIDENCE=2` caps repeats on top. Over-sampling to hit a big row target is exactly the overfitting failure mode in §12.

**Measured envelope (observed):**

| | value |
|---|---|
| build+image per scene (`SKIP_QA=1`) | ~2 min → 204 scenes ≈ **~7 h** |
| retained disk per scene | ~20 MB (`build_objects` 8.9 GB / 446) |
| binding constraint | **wall-clock**; transient raw-build peaks pressure free disk |

Driven by the `seismic-gen` systemd unit (env: `SKIP_QA=1 BUILD_CONCURRENCY=1 IMAGE_GEN_CONCURRENCY=1 NLI_DEVICE=cpu MemoryMax=8G TARGET_SCENES=450 QUEUE_AHEAD=3`), monitored by `scripts/gen_monitor.sh` (§11.1). Generation ran with QA skipped; **QA is the remaining step.**

### 13.1 QA pass — complete

**Scope: new-only, clean regen.** QA ran over the **204 new diverse scenes** (`graphs/_new_only/`, 408 graphs × 2 views), *not* the 446 total. The stale Jul-18 `verified_qa.jsonl` was backed up (`verified_qa.PRE_REGEN_0726.jsonl`). Mechanics in **§8.4**.

**Result: complete.** All 5 shards exited cleanly → **1,847 rows (1,819 after `row_id` dedup), 204 / 204 scenes** (full coverage), balanced across the 8 scene types. `qa_shards.sh finalize` merges the shards → `verified_qa.jsonl` → CSV.

**Run shape (5 sharded workers, resume-capable — §8.4), tuned across the run:**

| knob | final value | why |
|---|---|---|
| workers | 5 (`seismic-qa-0..4`) | ~1.7 GB RAM each on the 15 GB box; a transient 0 G spike was survived (per-worker `MemoryMax` contains a runaway, resume recovers it) |
| evidence | **plain (no value tags)** | 2.2× throughput — below |
| `QUESTION_PER_GRAPH` | **5** | sized to the plain-evidence rate so 5 workers cover **all 408 graphs in ~6 h** — full diversity, ~2 k rows, no idle. Q sets *spread*, not total (throughput-limited): Q=12 reached only ~60 scenes/6 h (concentrated), Q=3 finished in ~3.5 h (under-used the window) |
| `CANDIDATE_PER_QUESTION` | 3 | trimmed from 5 |
| sglang context | 4096 | richer evidence overflowed 2048 (§8.1) |
| NLI | CPU, `OMP_NUM_THREADS=2` | GPU held by sglang; don't oversubscribe cores |

**Throughput — value-tag removal was the unlock.** Tagged evidence ran ~157 rows/hr (5 workers); **plain evidence ~330–380 rows/hr (2.2×)** because prompts got shorter — that is what let Q climb back to 5 (full 6 h coverage). NLI-on-CPU + the 5-worker RAM cap are the hard walls.

**Class balance — one axis was off, now fixed.** Scene-type is balanced by design (weighted `ratio_per_types`). But **object class** skewed **onlap ~44%** of regions (fault 27 / closure 17 / salt 11) — the onlap-as-section-doc bug (§8.2 step 1), now **fixed at source**. For datasets built *before* the fix, **`qa_shards.sh balance`** auto-caps the top class by dropping its lowest-collateral (onlap-only) rows first (verified: onlap 864→512, others untouched, 1,467 rows).

**Composition — the meta-limit.** The pipeline controls **correctness** (the gate stack), not **composition**, so the dataset inherits whatever the 1.5B naturally emits, filtered for truth: it came out **~75% single-object / ~9% genuine 2-object comparison / ~94% quantitative** — emergent, not steered. The generation-side fixes (onlap-as-object; multi-object rule + same-type pairing, §8.2) shape that mix *at the source*; the verification side is now sound (NLI label bug fixed, guards proven necessary, §8.3). Interpretive/qualitative answers already exist (fault patterns, trap types, fluid class; ~5%) — a frequency dial, not a missing capability.

**Quality fixes landed** (all keep the gates): **NLI label-order bug fixed + entailment required** (§8.3); cross-type mismatch 25%→0% and attribute mismatch ~12%→0 (guards, §8.3, *proven irreplaceable by NLI*); compound half-answers fixed + multi-object comparison rule (§8.2); plain evidence + `<SEG>` owned by DatasetMaker (§8.1/§9); mask-attribute code refactored to `compute_attribute.py` (§6.3).

**Finalize:** merge `verified_qa_shard_*.jsonl` → `Dataset/verified_qa.jsonl`, then `python Dataset/DatasetMaker.py` builds the CSV (regions/masks + evidence `<SEG>`). Launch/resume with `scripts/qa_shards.sh`; sglang via `scripts/run_sglang.sh` (check `nvidia-smi` first — training reclaims the GPU per fold).

---

*Generated as a code-grounded walkthrough of the repository. File references point at the current implementation; regenerate graphs/builds after any change to the extraction or transform layers, since those stages only affect newly built artifacts.*
