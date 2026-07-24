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

The non-negotiable design property: **truthfulness**. Answers are grounded in `parameters.db` values and mask-measured geometry, gated by **NLI coverage** (every fact entailed, trust ≥ 0.7), a question-coverage **edge gate**, and a coordinate-based entity-swap guard, so the model learns real correspondences instead of plausible-sounding fiction.

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

1. computes per-category counts from the ratios (leftover rounding assigned round-robin),
2. writes one JSON **build config** per sample to `build_configs/{category}_{uuid}.json`,
3. writes a **recipe** `recipes/recipe_N.yaml` listing the sample names + the ratio plan.

**Current setting:** `sample_types = ["fault_complex"]` (all others commented out) with a `[100,100,500]` (LOW) cube, and **`sample_population: 2`** — i.e. **one recipe = 2 samples**, deliberately small. Scale is reached by **calling `populate()` repeatedly** (the driver in §11.1 tops the queue up), not by one huge recipe: a shallow queue keeps the disk footprint and the blast-radius of a bad config small, and lets the run stop cleanly at any point.

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
- **salt / onlap** — connected components named by order.

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

For each view (inline, crossline) it **deep-copies** the properties graph (the DB graph stays pristine) and overlays scene geometry from `scene_position.json`: it attaches `x`, `y`, `bbox` extents, `color`, and `view` to matching nodes, and **adds visual-only nodes** (with `HAS_VISUAL_OBJECT` edges) for objects the DB graph didn't already have. Broad/noisy visual objects (`onlap` components, `lithology`, `age_depth`) are excluded from object-level QA. Output: `graphs/properties_2d_graph/{sample}_{view}_properties_2d_graph.json`, which also carries the shared `scene` block (image/overlay paths).

**View-filter (why the per-view graph is pruned).** A 3D fault need not intersect every 2D slice: the DB knows 5 faults, but an inline section may render only 4. The per-view graph files existed, but their *contents* were identical and unfiltered — so RAG could serve `fault_2`'s facts on a view where `fault_2` isn't in the picture, generating QA about an object with **no mask** (which DatasetMaker could only log as `[NEG-NAMED]` and emit as a bogus maskless "negative"). `_copy_graph_with_2d_positions` now **prunes object instances that have no position in this view** (`positions` is exactly the view's rendered objects) and drops edges touching them. Section/hub nodes (the category node, the type hubs) never carry a per-view position and are always kept.

**View-scoped recount.** Pruning instances would otherwise leave the category node claiming the DB total ("7 faults" over a section showing 6), so `number_faults` is **recomputed from the surviving instances**. `number_fault_intersections` and `fault_mode` are *not* recomputed: they describe the 3D arrangement, not a per-slice count, and there is no per-view intersection geometry to derive them from. `number_hc_closures` is also left alone — it is an HC *subset*, not a plain instance count (a gap to close before enabling closure/salt-heavy categories).

### 6.3 Mask-measured visual attributes (dip, area)
Also in `properties_2d_graph.py` (`_mask_features` / `_dip_degrees`), computed **at 2D-graph time straight off each object's scene mask** — so the numbers reflect what's actually visible in pixels, not an opaque simulator knob:

- **fault `dip_deg`** — apparent dip = the angle of the fault trace's dominant line, 0°=flat … 90°=vertical, magnitude only. Estimation is **RANSAC + inlier gate**, not plain PCA: RANSAC finds the dominant collinear pixels and the angle is PCA-fit on those *inliers*, so a crossing structure or fragmented trace can't drag a moment-fit flat (which produced geologically absurd near-horizontal fault dips). A mask whose dominant line covers **< `_DIP_MIN_INLIER_FRAC` (0.5)** of the pixels returns `None`: on the now-individual per-object masks a low inlier fraction means the fault's *own* trace is non-planar (listric/branching) or fragmented — and a non-planar fault has no single dip (its pattern is still carried by the graph's `fault_mode`). On a clean single-fault mask every pixel is an inlier, so it reduces exactly to the old PCA angle. Deterministic (fixed RANSAC seed).
- **closure / salt `area_pct`** — mask coverage as a percentage of the section.

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

- **`ALLOWED_PROPERTY_EDGES`** gates which facts become evidence: model counts (`number_faults`, `number_hc_closures`, `number_fault_intersections`), `salt_inserted`, `fault_mode`; fault `throw` + `dip_deg`; closure `fluid`, `intersects_*`, `area_pct`. (Sub-seismic `shear_zone_width`/`gouge_pctile` and the opaque `tilt_pct` are intentionally excluded.) `fault_mode` is further **whitelisted** to genuine geological patterns (`relay_ramp`, `horst_and_graben`, `branching`, `staircase`); `random` is a Synthoseis generation setting, not a structure, so it emits **no pattern sentence** rather than leaking the simulator into the evidence.
- **Objects are named by COORDINATE, not "Type N".** `node_name` renders each numbered object as `"the <type> at [x,y]"` using its center (from the position relations, via a per-view coord map) — e.g. `"the fault at [57.5,289]"` — for **every** object type. The internal id (`fault_2`) is unchanged (mask routing, `object_id`, linkage all still key on it); only the *displayed* reference changes. This lets the LM reason spatially ("the closure at [22,140] is left of the fault at [57.5,289]") because the position is in the text. The coord in the name is **plain** (no `<bbox>`/`<center>` tag — the current model doesn't consume that); the tagged center still comes from the separate "sits near `<center>[x,y]</center>`" line as the grounding value. (This replaced the old `Fault N` = internal_index+1 naming.)
- **Templates** render each: `"the fault at [57.5,289] has throw of about <nums>124</nums> ms"` (ms TWT, §6.1), `"… dips at about <nums>58</nums> degrees"`, `"the closure at [22,140] covers about <nums>12</nums> percent of the section"`, count/boolean/intersection phrasings, and grouped **position** (`<center>[x,y]</center>`) / **extent** (`<bbox>[…]</bbox>`) sentences. The whole reference is wrapped in `<object>…</object>`.
- **Derived readings (tier-2).** Beyond the raw facts, `TextTransform` emits deterministic, **source-backed geological readings as additional grounded evidence lines** sharing the object's `object_id` (so they mask the same object and route identically): `dip_deg` → `"appears steeply / moderately / gently dipping in this section"` (30/60° descriptive scale, scoped to *apparent* dip), `fluid` oil/gas → `"hydrocarbon-bearing closure"` / brine → `"water-bearing closure"`, `intersects_fault` → `"fault-dependent closure"` (grounded in Synthoseis's own closure-type logic, `Closures.py:1562`), `intersects_onlap` → `"onlap trap"`. Because they're real evidence, they are **retrievable and NLI-checkable exactly like the raw facts** — a question/answer may use the geological term and still clear the gates. Magnitude labels (large/small throw, big closure) are deliberately *not* derived: no sourced numeric cutoff exists, so they'd be arbitrary. This gives three grounding tiers: raw measured fact → deterministic derived reading → interpretive reasoning (§8, un-checked, prompt-bounded).
- **Special tokens** `<object>`, `<nums>`, `<center>`, `<bbox>` mark the spans that carry entity/numeric identity. They live in the **evidence** (which keeps them as ground truth), not in the answer (which is plain text, §9.2). They serve two roles:
  1. **the model's regression targets** — training preprocessing substitutes each tagged evidence span with a special token and takes the value as the target (§9.2, §9.4),
  2. **stripped before embedding and before NLI** (§7.3, §8.3), because they are noise to both.
  Fidelity is enforced by the prompt rule "keep every value/coordinate exactly as the Evidences give it" (§8.2) + the coordinate entity guard (§8.3), not by tag-copying.

Each evidence item keeps its structured fields (`object_id`, `edge`, `target`, `relation`, `trace_type`) so masks can be re-associated later.

### 7.3 RAG construction & retrieval
**Stage owner:** `Verifier/create_rag.py` (`Rag`)

- **Documents are OBJECT-LEVEL, not per-fact.** `prepare_content` groups every sentence by `object_id` into **one `Document` per object** (`page_content` = all its sentences joined), with the per-fact structure preserved in `metadata["facts"]` = `[{edge, target, text, trace_type}]`. Model-level facts group under the category node as a "section" pseudo-object. **Why:** per-fact documents fragmented one object across many docs, so a narrow query fanned out and pulled sibling objects' facts (pollution). With object-level docs, a query finds a *coherent object*, verification checks each answer fact against that object's whole fact set (§8.3), and masking stays one-object-one-region — while `metadata["facts"]` keeps evidence/mask routing per-fact.
- **Vector store:** `InMemoryVectorStore` with HuggingFace `all-MiniLM-L6-v2` embeddings, wrapped in **`_TagStrippingEmbeddings`**.
- **Tag-stripping embedder (both sides).** Evidence tags (`<object>`, `<nums>`, `<center>`, `<bbox>`, and the segmentation marker `<SEG>`) are *noise* to the embedder. Embedding them dropped same-fact similarity from ~1.0 to **~0.62**, so a tagged doc vs the LLM's untagged `RETRIEVAL_QUERY` scored *below* `MIN_RETRIEVAL_SCORE` and got rejected. The wrapper strips tags before encoding **documents and queries**; stored `page_content` keeps its tags, so nothing downstream changes.
- **Graph retrieval:** `langchain_graph_retriever.GraphRetriever` with an `Eager` strategy (`start_k=6, k=20, select_k=20, max_depth=3`) traversing metadata edges (`object_id↔object_id`, `object_id↔parent_id`, `parent_id↔category_id`, `source↔parent_id`). The `edge↔edge` link was **deliberately removed** — it used to fan a narrow seed out to every object sharing a relation (all closures with `intersects_fault`), collapsing precision. **Breadth now comes from the query, not from cross-object fan-out.**

So retrieval isn't circular self-lookup: a query embeds, hits its nearest evidence docs, then walks a bounded neighborhood of the graph around them.

---

## 8. LLM, prompts, and QA generation

**Stage owners:** `Verifier/llm_machine.py` (prompts + model), `Verifier/generator_pipeline.py` (`RagWorkflow`, the loop)

### 8.1 The model
A **local vLLM/sglang** server (`http://localhost:8000/v1`) running `Qwen/Qwen2.5-1.5B-Instruct`, driven via `langchain_openai.ChatOpenAI`. Two tuned client bindings: **question** (higher temp/penalties for variety) and **answer** (low temp for faithfulness). Outputs are Pydantic-parsed JSON (`QuestionBatchStructure`, `AnswerBatchStructure`) with a strict output contract and retries. (The `reason` client/parser/prompt were removed with the reason column.)

The shared `MASTER_PROMPT` sets a **senior seismic interpreter (structural geologist) persona** — a natural interpreter's voice, not a data readout — and enforces the truthfulness contract: only use the evidence; never invent objects/values/causes; never mention graph/metadata/synthetic/prompt/verification.

**Wording is now free; only values/coords are pinned.** The model may reword sentences and choose how to present a coordinate (`[x,y]`, `(x,y)`, "near x, y") — this is deliberately relaxed for variety and to let spatial comparisons read naturally. The one hard rule kept: *keep every numeric value and coordinate exactly as the Evidences give it (never invent, round, or change a number); always write a number as a **digit**, never a word ("2 faults", not "two faults"); and refer to each object by its coordinate as the Evidences do.* Answers no longer carry grounding tags at all (see §9.2 — the answer is emitted as plain text), so there is nothing for the model to tag; `RETRIEVAL_QUERY` stays verbatim (below) because verification looks it up.

**`RETRIEVAL_QUERY` is a verbatim copy, not a paraphrase.** Both prompts require the exact Evidence line(s) a question/answer rests on, one per line. A paraphrased query embeds differently *and* mis-attributes at verification; copying is also easier for a small model than rewording. Compound (multi-fact, multi-object) questions and answers are explicitly allowed, because coverage verification (§8.3) checks each fact independently.

### 8.2 The QA loop (per 2D graph, per view)
The loop aims for `QUESTION_PER_GRAPH` (**12**) passing rows, capped at `MAX_ATTEMPT` (`3×` that) outer attempts — right-sized to how few facts one 2D section actually carries, so it terminates early instead of grinding a fixed 200. For each graph `RagWorkflow.generate_for_graph`:

1. **Seed evidence.** `evidence_seeds` shuffles the objects and yields, per seed, **`OBJECTS_PER_SEED` object docs (default 2) + the section doc** — so a batch shows the generator several coordinate-named objects at once (enabling multi-object / spatial-relational questions) *and* the section-level counts/mode. Set `OBJECTS_PER_SEED=1` for the old single-object behaviour; the section doc always rides along (the attaching-the-section-doc fix cured a bug where a lone-object seed made count answers collapse to "1 fault"). The `generate_for_graph` loop re-creates the generator on exhaustion, so each cycle re-shuffles into fresh object groupings.
2. **Generate questions.** `question_batch_generation` produces natural GroundVQA-style questions (no tags, no leaked values), rotated across five **facets** — *presence/featureless, count, location, orientation/geometry, relationship between two named structures* (`QUESTION_FACETS`, shuffled per batch) — but *evidence-gated*: an angle the evidence can't support is skipped. Questions may be **simple or compound**, and the prompt now **prefers segmentation/localization, multi-object questions** ("segment the fault at [x,y] and the closure to its left") — which stay retrievable because they name only objects the Evidences describe. Multi-object is fed by the seed (`OBJECTS_PER_SEED`, step 1). Each question ships a `RETRIEVAL_QUERY` = the **verbatim** Evidence line(s) it rests on.
3. **Retrieve for the question.** `retrieve_many(retrieval_query)` runs each query line through the graph retriever, dedups, and keeps only docs with `_similarity_score ≥ MIN_RETRIEVAL_SCORE (0.7)`. (0.7, not 0.9: MiniLM cosine rarely clears 0.9 for related-but-not-verbatim sentences; precision is still enforced by NLI coverage + coordinate entity guard + edge gate below, so retrieval only governs candidate **recall**. Tag-stripped embedding (§7.3) is what makes 0.7 reachable at all.) No docs → reject the question.
4. **Generate answers.** `answer_batch_generation` returns up to `CANDIDATE_PER_QUESTION` (**5**) candidate answers, each with its own concise `RETRIEVAL_QUERY` (the claim, not the prose). For a localization/segmentation question the answer places a **`<SEG>` token after each object it localizes** (one per object, in order) *while still stating the grounded facts* — so it verifies through the same NLI path (the `<SEG>` is stripped before NLI/embedding, §7.3/§8.3, and never replaces a claim). Only the single best survives verification, so 5 (not 100) spreads phrasings without wasting generation/retrieval/NLI, and fits `max_tokens`.
5. **Ground + verify each answer** (`best_answer`) — see §8.3.
6. **Dedup by evidence.** The same evidence set may back at most `MAX_ROWS_PER_EVIDENCE (2)` rows, so identical images don't over-repeat. The key is now the **union** fact-set signature, so a throw+dip answer and a dip-only answer count as different grounding instead of colliding.
7. **Append the row** to `Dataset/verified_qa.jsonl` (dedup by `row_id`, atomic append + flush). Each row records question/answer/evidence, the verification score, `trace` (question/answer evidence), and `metadata` (graph path, view, scene image path, `graph_mtime`).

A `[TALLY]` per graph reports where attempts die (question reject / answer reject / row skip / passed) so you can see whether `MAX_ATTEMPT` is the bottleneck.

### 8.3 Verification: coverage, not a single entailment

The old design ran `filter_docs_by_trust` (one NLI check of the *whole answer* against each doc) plus a final `verify_answer`. That breaks on compound answers: a single-fact doc only partially matches a two-fact answer, hovers at the threshold, and facts silently lose their grounding. It is replaced by **`cover_answer`**:

- **Shared fact pool.** The pool is every fact of every retrieved object — the **question's docs *and* the answer's** (`object_docs = dedupe(question_docs + answer_docs)`). A fact the *question* retrieved can therefore ground the answer.
- **Coverage (forward pass).** The answer's `RETRIEVAL_QUERY` is already one fact per line, so it is split on `\n`; **every line must be entailed** by some pool fact at trust ≥ 0.7, and is attributed to the best-entailing one. Any uncovered line → reject. (An earlier attempt to sentence-split the *answer* instead was reverted: it broke decimals — `80.5` → `80`,`5` — and stripped subjects, causing 14 false rejects.)
- **Forward vs. reverse (order matters).** Forward runs first *and is the gate*: if any declared `RETRIEVAL_QUERY` line is uncovered, `cover_answer` returns `None` and the reverse pass never runs. Reverse only runs on an already-passing answer and can only **add** facts, never reject — forward = *truth of declared claims*, reverse = *completeness of the compound*.
- **Compound completeness (reverse pass).** The small model often writes *one* query line for a two-fact answer, leaving the second fact real but ungrounded. So each pool fact of the object(s) the answer **cites by coordinate** is checked in reverse — `response=fact, sources=[answer]`, i.e. *"does the answer assert this fact?"* — and entailed facts are attached. Object identity is the coordinate (`coord_refs`), so only facts sharing the answer's cited coords enter the reverse pool (a different object's facts can't leak in). NLI decides; **no answer-splitting, no number-parsing**. Result: multi-evidence rows went **0 → 20/47**.
- **The NLI score itself** is a *hybrid* verifier (`longtracer`): bi-encoder semantic similarity (`avg_score`) gates a cross-encoder 3-class NLI (contradiction/neutral/entailment). A claim is supported when `avg_score ≥ 0.40` **or** `entailment > 0.5`, **and not** `contradiction > 0.5`; the thresholded `trust_score` is the similarity, the NLI drives the PASS/contradiction gate. The pipeline requires `PASS ∧ trust ≥ 0.7`. The contradiction guard is what rejects a wrong-valued fact ("dips 40°" when the answer says 65.8°).
- **Tag-stripped NLI.** Like the embedder, NLI compares tag-free text (`<object>`/`<nums>`/`<center>`/`<bbox>`/`<SEG>` all stripped). A true throw fact scored **0.616 tagged vs 0.823 stripped** — tags alone pushed real facts below threshold. So a segmentation answer's `<SEG>` markers don't affect its verification. Stored evidence/answer keep their tags.
- **Entity-swap guard** (`answer_objects_in_docs`) — now **coordinate-based**. Since objects are named by coordinate, every coordinate group the answer cites (center `[x,y]` or box `[x1,y1,x2,y2]`, any bracket style, number-normalized, matched as *whole groups* so a box is never mistaken for two centers) must appear in the evidence. A coord not in the evidence is a swap/fabrication → reject; NLI still judges the *relation* on top ("A at [..] is left of B"). (This replaced the old `<object>`-string + "Type N" match, which coord-naming and free rewording made unworkable.)
- **Question-coverage edge gate** — the answer must address what the question *asks*, not merely be grounded. This replaced the old keyword *facet* gate: both the question and the answer are grounded to per-fact evidence carrying `metadata["edge"]`, so the gate compares **edge sets** directly — the answer's grounded edges must **cover** the question's grounded edges. A compound question grounds to ≥2 edges (e.g. `dip_deg` + `number_faults`), so an answer that drops a clause is missing that edge and is rejected — precisely the failure the keyword any-overlap facet gate let through. When the question grounds to nothing, it passes (lenient, no false-reject).
- **Verdict:** `{"verdict": "PASS", "score": mean(covered trust)}`. Best-scoring surviving answer wins.

The **question** is also grounded per-fact (`cover_answer(..., require_all=False)`, lenient — question evidence is additive, not a gate), and the row's evidence is the **union of question ∪ answer facts**. This both completes the grounding (a comparison question keeps the objects it compares) and *repairs* some answer-side mislabels: for "how many fault intersections?", the answer sometimes grounds `number_faults` while the **question** correctly grounds `number_fault_intersections` — the union carries the right fact.

> **Why this design:** questions and answers come from the *same* small model but must survive **retrieval gating + NLI coverage + coordinate entity guard + edge gate** against graph-derived evidence. The graph is the source of truth; the LLM only phrases. Note the honest boundary: verification checks the answer against **the facts its own `RETRIEVAL_QUERY` fetched**, so a *self-consistent* wrong answer can pass (§12).

---

## 9. Dataset making

**Stage owner:** `Dataset/DatasetMaker.py`

Reads `verified_qa.jsonl` and emits `Dataset/multimodal_multi_image_dataset.csv` (and a jsonl).

### 9.1 Regions and the row mask

1. Load the shared scene image + its objects from `scene_position.json`.
2. **Split section-scoped from object-scoped evidence.** `SECTION_EDGES` (= the `EDGE_TYPES` keys: `number_faults`, `fault_mode`, `number_fault_intersections`, `salt_inserted`, `number_hc_closures`) describe the **whole section**, not one object. Only object-scoped facts drive the per-object loop.
3. **Match evidence to regions** (`evidence_matches_region`): an evidence item lights up an object region when its `object_id` matches, or its type matches a type-global region, or it's an edge-type fact for that class. The subtle rule: object-specific evidence like `fault_0` falls back to the **type-global** mask *only when that object has no individual mask* (`individual_ids` guard) — so "dip of fault 1" masks fault 1 if a per-fault mask exists, else the all-faults mask, but **never** bleeds onto `fault_1` from `fault_0` evidence.
4. **Section facts → ONE whole-section region.** Previously a section fact matched *every* fault, so the per-object loop stamped an **identical** `<region>` block onto each one — 5 byte-identical blocks, all pointing at the same `mask_idx=0`, i.e. *one mask described five times*, teaching the VLM that each individual fault "is 17 intersections". Now section facts are emitted **once** as a single region (`object_id="section"`, union bbox over the referenced type, mask still covering all of them), so the (text → segment) pairing is 1:1 like every object row.
5. **One mask per region** (`build_region_mask`), **not** a single composite. Each region gets its own binary PNG under `Dataset/masks/` (keyed `_r{idx}_`): an object region = that one object's mask; the whole-section region = the **union** of every object of the referenced type. A region is only emitted if its mask has real pixels. So one row is **1 image → N masks → N evidence blocks → N `regions`**, all index-aligned: `masks[i] ↔ regions[i] ↔ the i-th `<region>`/`<SEG>``. Each region's `mask_idx` points at its own mask. (Verified against real data: region `bbox`/`center` reproduce the mask pixels exactly, and evidence `<center>`/`<bbox>` values equal the region's.)
6. **No-evidence rows fall back to the DEFAULT SHARED SCENE.** A featureless / section-only row (no object region built) no longer goes maskless. It's emitted with the scene image + a **whole-scene mask** — the union of every object, or a full-frame mask if there's nothing to outline (`_default_scene_mask`) — as **one global region** with its own `<SEG>`. So *every* row carries image + mask + region + `<SEG>` (the empty-mask negative is gone: 0 maskless rows, was ~142). The global region's `object_name` is `"the section"` (whole-section reference); its seismic class lives in `object_type`/`class_id`, derived from the section fact's edge (e.g. `number_faults → fault`).

### 9.2 The answer is plain; the evidence carries the ground truth

**Both the answer and the evidence text are plain** — grounding tags are *unwrapped* (`_untag_answer`: markup stripped, **value words kept** so the text stays retrievable and human-legible). Answer: `<answer>…</answer>` around plain prose. Evidence: `<evidence>` wrapping one `<region>\n{plain facts}\n<SEG>\n</region>` block **per mask**, e.g. `Fault 4 dips at about 65.8 degrees.` — the number stays, the `<nums>` tag is gone. Invariant checked at build across all rows: `#<region> == #<SEG> == #</region> == len(masks) == len(regions)`, and each `<region>` holds exactly one `<SEG>`.

**The structured grounding moved to the `regions` column** (the tags' *job*, not the tags). Each `regions[i]` (one per `<SEG>`/mask) carries:
- **`object_name`** — the object's coordinate reference (`"the fault at [57.5,289]"`) for an individual object, or `"the section"` for a global/section region. This *replaces* `object_id` — the internal id is never surfaced. The seismic class is carried separately by `object_type`/`class_id`.
- **`seg_idx`** — ties this region to the i-th `<SEG>`.
- **`values`** — the machine-readable ground truth, split by **value type** (which is also the head split): `{"measure": {…continuous scalar magnitudes: dip_deg, throw, area_pct…}, "derive": {…counts / categories / booleans: number_*, fault_mode, fluid, salt_inserted, intersects_*…}}`. `measure` are **regression** targets, `derive` are **classification/count** targets. The rule is the value's type (magnitude vs count/category), not where it was fetched — so `throw` sits with `dip` in `measure` even though it comes from the DB. Both keys always present; the same values also read as words in the evidence text (retrievable).
- plus `class_id`, `bbox`, `center`, `mask_idx`, `object_type`, `view`.

> **Why values live in `regions`, not inline tags.** Unwrapping the evidence keeps it retrievable/readable while the *exact* regression targets sit in `regions[i].values` (deterministic, no parsing of free text). Slots/placeholders in the text were tried and reverted — pre-blanking destroys the ground truth. The dataset is the regression *target*; the model's training preprocessing maps a `<SEG>`/value to its `regions` entry.

Columns: `sample_id, images, masks, instruction, question, answer, evidence, regions`. (The former `question_regions` / `answer_regions` columns and the `region_grounded_variant` twin maker were **removed**.) `Dataset/upload_to_huggingface.py` pushes it to the Hub.

### 9.4 The grounding format & the model-side contract

The format follows **GLaMM/LISA**-style *inline* grounding (not GranD's sidecar). Where grounding lives, in the current schema:

- **Segmentation** → the `masks` + `regions` columns. `regions[i]` carries `{object_name, class_id, bbox, center, values, seg_idx, mask_idx, object_type, view}` for `masks[i]`, one per `<SEG>`. **Referring-segmentation is prompt-driven** (§8.2): the question prompt *prefers* casual, multi-object, spatial localization ("segment the fault at [x,y] and the closure to its left"), and the answer emits one `<SEG>` after each object it localizes. Crucially the answer **still states the grounded facts** — so it is verified by the *same* pipeline (retrieval + NLI coverage + edge gate + coord swap guard); `<SEG>` is a marker, not a substitute for a claim, and it is stripped before NLI/embedding (§7.3, §8.3) so verification is unaffected. `seg_idx` is the `<SEG>`↔region link the collator uses; a build-time `[SEG MISMATCH]` log flags any answer whose `<SEG>` count ≠ its region count.
- **Regressable values** → the tagged spans in the **`evidence`** column (`<nums>`, `<center>`, `<bbox>` around ground-truth values). The model's training preprocessing substitutes each tagged span with a special token and takes the value from the span as the **regression target** — the dataset carries the real value; the collator never asks the model to emit coordinate digits as free text.
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
- keeps a **shallow queue** (`QUEUE_AHEAD=6`) by calling `control_parameter` repeatedly — one recipe = **2 `fault_complex` samples** (`sample_population: 2`), rather than one huge recipe,
- stops at `TARGET_SCENES` or if free disk falls below `MIN_FREE_GB` (12 G),
- logs progress to `gen_driver.log`, build detail to `gen_run.log`.

**Measured cost:** ~**4.5 min/scene** (build + image gen + CPU QA) and only ~**13–33 MB/scene** retained. Disk is *not* the constraint (~250 scenes ≈ 6 GB); **wall-clock is** (250 scenes ≈ 19 h).

> **Restart caveat:** a watcher restart re-QAs already-built graphs and **appends**, over-sampling those scenes. During a long build the QA is therefore provisional — finish with **one clean QA pass** (reset `verified_qa.jsonl`, regenerate over all graphs) so every scene is sampled exactly once.

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
- **Only `fault_complex` / `fault_only` are proven.** Closure/salt categories are *structurally* supported (class ids, `EDGE_TYPES`, `CROSS_REFERENCE_EDGES`, `Closure N`/`Salt N` naming) but **untested**, and the per-view recount currently fixes **only `number_faults`** — so closure/salt-heavy categories will hit the same off-view count mismatch that was just fixed for faults. `onlap`/`lithology` are excluded by design (`EXCLUDED_VISUAL_OBJECTS`) and would yield only section-level/negative rows.
- **Excluded-object intermediates dominate build_objects.** `onlap/` renders can be ~62 MB of a ~65 MB sample even though onlap is excluded from the dataset — the retained cost per scene is otherwise only ~2.5 MB (scene images + fault masks). Not pruned (deliberately, no delete function); worth knowing if disk gets tight at large scene counts.

---

## 13. Scale & the current run

**Target:** **500 distinct images**. Each scene renders **2 views** (inline + crossline), so that is **250 scenes** — and scenes, not rows, are the unit that matters (§12: effective visual N ≈ distinct builds).

```
250 scenes × 2 views          = 500 distinct images
500 images × 12 QA (QUESTION_PER_GRAPH)   ≈ 6,000 verified rows
```

**Why 12 QA/image is the "optimal" pick:** it sits in the healthy VQA band (5–15 Q/image, cf. GQA/VQAv2) and well inside a `fault_complex` scene's fact budget (~5–7 faults × ~5 facts + section facts ≈ 30+), so the 12 questions land on **distinct facts** rather than rephrasing one — and `MAX_ROWS_PER_EVIDENCE=2` caps repeats on top. Pushing to ~45 QA/scene (what 20K rows over 250 scenes would need) would force over-sampling and is exactly the overfitting failure mode in §12.

**Measured envelope:**

| | value |
|---|---|
| build+QA per scene | ~4.5 min → 250 scenes ≈ **~19 h** |
| retained disk per scene | ~13–33 MB → 250 scenes ≈ **~6 GB** (of 52 GB free) |
| binding constraint | **wall-clock**, not disk |

Driven by `scripts/run_generation.sh` (§11.1), detached, with a hard stop at target or `<12 GB` free. Finish with **one clean QA pass** over all graphs (§11.1 caveat) to produce the final `verified_qa.jsonl` → CSV.

---

*Generated as a code-grounded walkthrough of the repository. File references point at the current implementation; regenerate graphs/builds after any change to the extraction or transform layers, since those stages only affect newly built artifacts.*
