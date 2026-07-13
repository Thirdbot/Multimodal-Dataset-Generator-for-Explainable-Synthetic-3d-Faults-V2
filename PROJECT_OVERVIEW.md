# simulationv2 — Project Overview

A local, end-to-end generator for a **multimodal seismic-interpretation dataset**. It synthesizes 3D seismic volumes with a physics-based simulator, turns each volume's ground-truth metadata into a knowledge graph, renders 2D sections + segmentation masks, and uses a graph-RAG + LLM + NLI loop to write **grounded, verified visual question–answer pairs** with pixel masks — the kind of data you fine-tune a seismic VLM on.

The whole thing runs on one machine, orchestrated by a set of asyncio file-watchers so that each finished sample flows through the pipeline independently instead of waiting for the whole batch.

---

## 1. Project goal

Train and evaluate **vision-language models on seismic interpretation** (fault detection, closure/salt identification, counting, localization, geometry). Real labelled seismic is scarce, proprietary, and expensive to annotate. This project manufactures an arbitrarily large, fully-labelled corpus where **every answer is traceable to a known simulation parameter or a measured pixel region** — no human annotation, no hallucinated geology.

Each dataset row is:

- one or more **seismic section images** (grayscale 2D slices),
- a **segmentation mask** highlighting exactly the object(s) the question is about,
- a natural **question**,
- a **verified answer** (NLI-checked against evidence),
- a short **chain-of-thought reason**,
- structured **regions** (bbox, center, class id, object id) for grounding.

The non-negotiable design property: **truthfulness**. Answers are grounded in `parameters.db` values and mask-measured geometry, gated by NLI entailment (trust ≥ 0.7) and an entity-swap guard, so the model learns real correspondences instead of plausible-sounding fiction.

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

The current `settings.yaml` + `control_parameter.py` are set to produce `fault_complex` samples with a `[100,100,500]` (LOW) cube.

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

### 6.3 Mask-measured visual attributes (dip, area)
Also in `properties_2d_graph.py` (`_mask_features` / `_dip_degrees`), computed **at 2D-graph time straight off each object's scene mask** — so the numbers reflect what's actually visible in pixels, not an opaque simulator knob:

- **fault `dip_deg`** — apparent dip = the angle of the fault trace's principal axis (PCA / covariance eigenvectors), 0°=flat … 90°=vertical. Near-round masks (eigenvalue ratio > 0.6) return `None`. Magnitude only (no left/right, to avoid image-convention mistakes).
- **closure / salt `area_pct`** — mask coverage as a percentage of the section.

These are attached as node attributes, so they flow through the tracer → text transform automatically, with no extra plumbing. This replaces the DB's opaque `tilt_pct` (dropped) and fills a gap the DB simply doesn't have (closure area). Only individual objects get them; the merged per-type mask (`object_id == type`) is skipped.

---

## 7. Graph → evidence → RAG

### 7.1 Tracing relations
**Stage owner:** `Tracer/tracer.py` (`EvidenceTracer`)

Walks the 2D graph and emits a flat list of **relations**: one per node property (`source, edge=key, target=value`) and one per edge (`source, HAS_*, target`), each with a `relation` triple and a raw `text`. This is deliberately exhaustive — it turns *every* surviving attribute into a candidate fact; the whitelist lives in the next stage.

### 7.2 Relations → natural evidence
**Stage owner:** `NaturalTransform/text_transform.py` (`TextTransform`)

Converts relations into short, inspectable English sentences — one sentence per triplet, no grammar variation. It's the **whitelist + renderer**:

- **`ALLOWED_PROPERTY_EDGES`** gates which facts become evidence: model counts (`number_faults`, `number_hc_closures`, `number_fault_intersections`), `salt_inserted`, `fault_mode`; fault `throw` + `dip_deg`; closure `fluid`, `intersects_*`, `area_pct`. (Sub-seismic `shear_zone_width`/`gouge_pctile` and the opaque `tilt_pct` are intentionally excluded.) `fault_mode` is further **whitelisted** to genuine geological patterns (`relay_ramp`, `horst_and_graben`, `branching`, `staircase`); `random` is a Synthoseis generation setting, not a structure, so it emits **no pattern sentence** rather than leaking the simulator into the evidence.
- **Templates** render each: `"<object>Fault 1</object> has throw of about <nums>124</nums> ms"` (ms TWT, §6.1), `"… dips at about <nums>58</nums> degrees"`, `"<object>Closure 1</object> covers about <nums>12</nums> percent of the section"`, count/boolean/intersection phrasings, and grouped **position** (`<center>[x,y]</center>`) / **extent** (`<bbox>[…]</bbox>`) sentences.
- **Derived readings (tier-2).** Beyond the raw facts, `TextTransform` emits deterministic, **source-backed geological readings as additional grounded evidence lines** sharing the object's `object_id` (so they mask the same object and route identically): `dip_deg` → `"appears steeply / moderately / gently dipping in this section"` (30/60° descriptive scale, scoped to *apparent* dip), `fluid` oil/gas → `"hydrocarbon-bearing closure"` / brine → `"water-bearing closure"`, `intersects_fault` → `"fault-dependent closure"` (grounded in Synthoseis's own closure-type logic, `Closures.py:1562`), `intersects_onlap` → `"onlap trap"`. Because they're real evidence, they are **retrievable and NLI-checkable exactly like the raw facts** — a question/answer may use the geological term and still clear the gates. Magnitude labels (large/small throw, big closure) are deliberately *not* derived: no sourced numeric cutoff exists, so they'd be arbitrary. This gives three grounding tiers: raw measured fact → deterministic derived reading → interpretive reasoning (§8, un-checked, prompt-bounded).
- **Special tokens** `<object>`, `<nums>`, `<center>`, `<bbox>` mark spans that must be **copied verbatim** by the LLM (enforced in the prompt) — this is how numeric/entity fidelity survives generation.

Each evidence item keeps its structured fields (`object_id`, `edge`, `target`, `relation`, `trace_type`) so masks can be re-associated later.

### 7.3 RAG construction & retrieval
**Stage owner:** `Verifier/create_rag.py` (`Rag`)

- **Documents:** each evidence sentence → a LangChain `Document`, `page_content` = the sentence, `metadata` = `{trace_type, source, object_id, parent_id, category_id, edge, target, relation}`. Parent/category ids come from `graph_hierarchy` so the retriever can walk the object→type→category structure.
- **Vector store:** `InMemoryVectorStore` with HuggingFace `all-MiniLM-L6-v2` embeddings.
- **Graph retrieval:** `langchain_graph_retriever.GraphRetriever` with an `Eager` strategy (`start_k=6, k=20, select_k=20, max_depth=3`) traversing metadata edges (`object_id↔object_id`, `object_id↔parent_id`, `parent_id↔category_id`, `source↔parent_id`). The `edge↔edge` link was **deliberately removed** — it used to fan a narrow seed out to every object sharing a relation (all closures with `intersects_fault`), collapsing precision. **Breadth now comes from the query, not from cross-object fan-out.**

So retrieval isn't circular self-lookup: a query embeds, hits its nearest evidence docs, then walks a bounded neighborhood of the graph around them.

---

## 8. LLM, prompts, and QA generation

**Stage owners:** `Verifier/llm_machine.py` (prompts + model), `Verifier/generator_pipeline.py` (`RagWorkflow`, the loop)

### 8.1 The model
A **local vLLM** server (`http://localhost:8000/v1`) running `Qwen/Qwen2.5-1.5B-Instruct`, driven via `langchain_openai.ChatOpenAI`. Three tuned client bindings: **question** (higher temp/penalties for variety), **answer** (low temp for faithfulness), **reason** (middle). All outputs are Pydantic-parsed JSON (`QuestionBatchStructure`, `AnswerBatchStructure`, `ReasonStructure`) with a strict output contract and retries.

The shared `MASTER_PROMPT` sets a **senior seismic interpreter (structural geologist) persona** — a natural interpreter's voice, not a data readout — and enforces the truthfulness contract: only use the evidence; never invent objects/values/causes; never mention graph/metadata/synthetic/prompt/verification; copy tagged spans exactly. The question, answer, and reason prompts are deliberately kept short (a handful of rules) so the 1.5B model complies.

### 8.2 The QA loop (per 2D graph, per view)
The loop aims for `QUESTION_PER_GRAPH` (**12**) passing rows, capped at `MAX_ATTEMPT` (`3×` that) outer attempts — right-sized to how few facts one 2D section actually carries, so it terminates early instead of grinding a fixed 200. For each graph `RagWorkflow.generate_for_graph`:

1. **Seed evidence.** `evidence_seeds` shuffles the docs and yields small per-object packets (so a batch is about one object's facts, reducing cross-object bleed).
2. **Generate questions.** `question_batch_generation` produces natural GroundVQA-style questions (no tags, no leaked values), rotated across **facets** (presence, count, location, orientation, relationship, **geological character/trap type**, and **bundled** two-property questions) but *evidence-gated* — an angle the evidence can't support is skipped. Because the tier-2 readings (§7.2) are grounded evidence, questions may ask geological character (steepness class, trap type, fluid) and may **bundle two or three related facts of one structure** into a single natural question (e.g. where it sits and what it meets). Each question ships a `RETRIEVAL_QUERY` (evidence-like sentences, **one line per bundled fact**), so a compound question retrieves each of its facts.
3. **Retrieve for the question.** `retrieve_many(retrieval_query)` runs each query line through the graph retriever, dedups, and keeps only docs with `_similarity_score ≥ MIN_RETRIEVAL_SCORE (0.7)`. (0.7, not 0.9: MiniLM cosine rarely clears 0.9 for related-but-not-verbatim sentences; precision is still enforced by the NLI trust filter + entity guard below, so retrieval only governs candidate recall.) No docs → reject the question.
4. **Generate answers.** `answer_batch_generation` returns up to `CANDIDATE_PER_QUESTION` (**5**) candidate answers, each with its own `RETRIEVAL_QUERY` (the claim(s), one line per fact). An answer may be **one or two sentences** so it can address a bundled question — stating each fact and, only where the evidence connects them, how they relate. This is safe because NLI verifies the answer **per-sentence** (§8.2 step 5): a two-fact answer passes iff *every* clause is grounded, so bundling raises richness without loosening truth (it does lower per-attempt yield — more clauses, more that must hold). Only the single best survives, so 5 (not 100) spreads phrasings without wasting generation/retrieval/NLI, and a 5-item batch fits `max_tokens`.
5. **Ground + verify each answer** (`best_answer`):
   - Retrieve on the answer's own claim, merge with the question's docs, dedup.
   - **NLI trust filter** (`filter_docs_by_trust`, longtracer `check_batch`): one entailment check per doc, thread-pooled in a single batch call, keeping only docs that *entail* the answer with trust ≥ 0.7. Empty → reject.
   - **Entity-swap guard** (`answer_objects_in_docs`): every object the answer names (tagged `<object>` **and** untagged "Type N") must actually appear in the grounding, so "asked Closure 10, answered Closure 8" is rejected even when NLI would wave through the near-duplicate wording.
   - **Final verdict** (`verify_answer`): NLI `check` of the natural answer against the grounding text → PASS/score. The best-scoring surviving answer wins.
6. **Dedup by evidence.** The same evidence set may back at most `MAX_ROWS_PER_EVIDENCE (2)` rows, so identical images don't over-repeat.
7. **Reason.** `reason_generation` writes a 2–3 step chain-of-thought that *justifies* the (already-verified) answer from the shared evidence. The reason is **not** NLI-checked (it's generated after verification), so it's the one place interpretive bridging lives — the prompt bounds it to the *definitional* geological reading (steep dip → steeply-dipping fault; fluid → hydrocarbon-bearing; closure meeting a fault → fault-dependent) and **forbids** unstated process (tectonic/depositional history, migration, seal, charge) and "could mean" leaps. Since the tier-2 readings (§7.2) are already grounded evidence, the reason mostly restates them in interpreter's voice rather than inventing.
8. **Append the row** to `Dataset/verified_qa.jsonl` (dedup by `row_id`, atomic append + flush). Each row records question/answer/evidence, the verification score, `trace` (reason + question/answer evidence), and `metadata` (graph path, view, scene image path, `graph_mtime`).

A `[TALLY]` per graph reports where attempts die (question reject / answer reject / row skip / passed) so you can see whether `MAX_ATTEMPT` is the bottleneck.

> **Why this design:** questions and answers are generated by the *same* small model but must survive **retrieval gating + NLI entailment + entity guard** against graph-derived evidence. The graph is the source of truth; the LLM only phrases and reasons. Verifying the *natural* answer (not the retrieval proxy) closes the loop.

---

## 9. Dataset making

**Stage owner:** `Dataset/DatasetMaker.py`

Reads `verified_qa.jsonl` and emits `Dataset/multimodal_multi_image_dataset.csv` (and a jsonl). Per row (`build_row`):

1. Load the shared scene image + its objects from `scene_position.json`.
2. **Match evidence to regions** (`evidence_matches_region`): an evidence item lights up an object region when its `object_id` matches, or its type matches a type-global region, or it's an edge-type fact for that class. The subtle rule: object-specific evidence like `fault_0` falls back to the **type-global** mask *only when that object has no individual mask* (`individual_ids` guard) — so "tilt/dip of fault 1" masks fault 1 if a per-fault mask exists, else the all-faults mask, but **never** bleeds onto `fault_1` from `fault_0` evidence.
3. **Composite the row mask** (`build_row_mask`): union only the **retrieved** objects' scene-registered masks into one binary PNG (large objects painted first) under `Dataset/masks/`. The row's single mask therefore highlights *exactly what the question pulled*.
4. Build `regions` (per object: `image_idx=0`, `mask_idx=0`, bbox, center, class id, object id) and an `<region>…<SEG>…</region>` evidence block.
5. **Negatives are kept**: a featureless/"no faulting" row has no object to outline, so it's emitted with `masks: []` and empty `regions` rather than dropped — valid VQA supervision for absence.

Columns: `sample_id, images, masks, instruction, question, answer, evidence, reason, regions`. `Dataset/upload_to_huggingface.py` pushes it to the Hub.

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
        │   graphs/properties_2d_graph/{sample}_{view}_..._2d_graph.json
        │        │  EvidenceTracer → TextTransform → Rag (embed + graph retrieval)
        │        │  RagWorkflow: question → retrieve → answer → NLI trust → entity guard → verify → reason
        │        ▼
        │   Dataset/verified_qa.jsonl
        │        │  DatasetMaker: match evidence→regions, composite row mask, keep negatives
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

---

## 12. Design rationale & known limitations

**Why grounded-and-verified instead of just prompting a big model:** the entire value proposition is *trust*. A seismic VLM trained on plausible-but-wrong answers learns nothing useful. Every number traces to `parameters.db` or a measured pixel region; NLI entailment + the entity guard reject drift; tagged spans force verbatim fidelity.

**Why measure geometry from masks (dip/area) instead of the DB:** the DB's `tilt_pct` is a simulator rotation *fraction*, not a legible angle, and it stores no closure area at all. A VLM can only verify what's in the pixels, so the visible geometry is measured from the same masks the model will see. The DB stays the oracle for *identity, counts, fluids, intersections*; masks own *visible geometry*.

**Honest limitations (also tracked in `README.md` TODOs):**

- **Single simulator ceiling.** Synthoseis defines the visual domain; expect a domain gap to field seismic. This is pretraining/augmentation data, not a replacement for real-data fine-tuning.
- **Text-mediated verification of a visual task.** QA is generated and NLI-checked against *text* evidence derived from ground truth; it asserts the answer is *true of the scene*, not that it's *visually inferable* from that particular 2D section. Mask-measured dip/area narrow this gap; it doesn't fully close it.
- **Small generator model.** Qwen2.5-1.5B is fast and cheap but a weak writer; the gates catch errors but also throttle yield (watch the `[TALLY]`).
- **Per-fault masks depend on the wrapper.** If `faults/fault_XX.zarr` isn't emitted (guard off / older builds), faults fall back to a single merged mask and lose per-fault dip/localization.
- **Combinatorial masks ≠ visual diversity.** Subset-masks from one complex scene are correlated; effective visual N ≈ **number of distinct builds**. Split train/val **by build** to avoid leakage, and scale the *build* count for generalization.

---

*Generated as a code-grounded walkthrough of the repository. File references point at the current implementation; regenerate graphs/builds after any change to the extraction or transform layers, since those stages only affect newly built artifacts.*
