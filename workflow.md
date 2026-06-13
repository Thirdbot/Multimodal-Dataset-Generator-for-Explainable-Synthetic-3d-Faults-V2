# SimulationV2 Workflow

This project builds a controlled synthetic seismic dataset pipeline. The core idea is to use Synthoseis to generate 3D seismic examples, extract DB-grounded metadata into graphs, generate 2D image views and masks, then use graph evidence to produce verified natural-language rows for a future multimodal/VLM dataset.

The important design rule is:

**The graph is the factual source. Images are visual grounding. Do not infer new geological relations from 2D image overlap.**

2D slices are projections of a 3D volume. A bbox intersection in one image view can be misleading, so image metadata should not rewrite geological facts such as “fault cuts closure” or “salt intersects closure”. Those semantic relations should come from `parameters.db` / extracted graph evidence.

## Main Stages

1. **Recipe and config generation**
   - Recipes describe what kinds of samples should be built.
   - Configs are generated from recipes.
   - The intended category, such as `fault_only`, `salt_fault_mixed`, `onlap`, or `full_mixed`, controls which parameters matter.

2. **Synthoseis build**
   - Synthoseis builds the 3D model and writes outputs under `builds/<sample_id>`.
   - Important generated files include seismic volumes, closure masks, fault masks, onlap masks, lithology arrays, depth/age arrays, and `parameters.db`.
   - Fault individual masks are captured through a wrapper during build, not by editing Synthoseis internals.

3. **Low-level DB extraction**
   - `scripts/low_level_tracer.py` reads `parameters.db`.
   - It extracts tables such as:
     - `model_parameters`
     - `fault_parameters`
     - `closure_parameters`
   - The output is a DB-extracted JSON used as raw metadata.

4. **Properties graph generation**
   - `scripts/graph_generator.py` and `scripts/graph_system.py` turn DB extracts into graph JSON.
   - Output:
     - `graphs/properties_graph/*_properties_graph.json`
   - This is the main factual graph.
   - It is filtered by sample category using `CATEGORY_FILTERS`.

5. **2D image and mask extraction**
   - `scripts/images_generator.py` reads a properties graph and the matching build folder.
   - It creates global and local images under:
     - `build_objects/images/<sample_id>/`
   - For each object type it can write:
     - `inline.png`
     - `inline_mask.png`
     - `inline_overlay.png`
     - `crossline.png`
     - `crossline_mask.png`
     - `crossline_overlay.png`
     - `timeslice.png`
     - `timeslice_mask.png`
     - `timeslice_overlay.png`
   - It also writes per-object-type position metadata:
     - `fault_object_position.json`
     - `closure_object_position.json`
     - `salt_object_position.json`
     - `onlap_object_position.json`
     - `lithology_object_position.json`
     - `age_depth_object_position.json`

6. **2D properties graph projection**
   - `scripts/properties_2d_graph.py` copies `properties_graph` and updates only matching object positions from the image metadata.
   - Output:
     - `graphs/properties_2d_graph/*_inline_properties_2d_graph.json`
     - `graphs/properties_2d_graph/*_crossline_properties_2d_graph.json`
     - `graphs/properties_2d_graph/*_timeslice_properties_2d_graph.json`
   - It only updates nodes that already exist in the source graph.
   - It does not add visual-only objects.
   - It replaces 3D position fields with 2D fields:
     - `x`
     - `y`
     - `x_min`
     - `x_max`
     - `y_min`
     - `y_max`
     - `view`

7. **Tracing and text evidence**
   - `Tracer/tracer.py` loads graph JSON and turns nodes/edges into relation records.
   - `NaturalTransform/text_transform.py` turns those relations into natural geological evidence.
   - It supports:
     - 3D position: `x0`, `y0`, `z0`
     - 3D extent: `x_min`, `x_max`, `y_min`, `y_max`, `z_min`, `z_max`
     - 2D position: `x`, `y`
     - 2D bbox: `x_min`, `x_max`, `y_min`, `y_max`
   - It intentionally does not say “inline”, “crossline”, or “timeslice” in the natural sentence.

8. **RAG / question-answer generation**
   - `Verifier/create_rag.py` converts graph evidence into LangChain `Document` objects.
   - `Verifier/llm_machine.py` generates questions and answers from evidence using a local vLLM-compatible endpoint.
   - `Verifier/generator_pipeline.py` is the newer hybrid RAG workflow.
   - It reads from:
     - `graphs/properties_2d_graph`
   - It writes:
     - `Dataset/hybrid_verified_qa.jsonl`

## Key Files

### `scripts/graph_generator.py`

Builds properties graphs from DB extracts. The important structure is `CATEGORY_FILTERS`.

This controls which DB tables and keys are allowed into each category graph.

Example categories:

- `boring`
- `fault_only`
- `fault_complex`
- `salt_only`
- `salt_fault_mixed`
- `onlap`
- `depositional`
- `full_mixed`

This is where graph content is intentionally restricted. If a field is not in the filter, it will not appear in the graph or text evidence.

### `scripts/graph_system.py`

Turns filtered DB rows into a graph.

Important behavior:

- Adds a category node such as `category:fault_only`.
- Adds object-type nodes such as `fault` or `closure`.
- Adds realized object nodes such as `fault_0`, `closure_0`.
- Skips invisible faults using `fault_voxel_count_list`.
- Reindexes visible faults so graph IDs stay continuous.
- Stores `original_fault_index` for fault wrapper matching, but this is filtered out of text evidence.

### `scripts/images_generator.py`

Extracts 2D images and masks from generated 3D arrays.

Current object sources:

- Fault:
  - `faults/fault_*.zarr`
  - `fault_segments_*.zarr`
  - `fault_intersection_segments_*.zarr`
- Closure:
  - `closures/oil.zarr`
  - `closures/gas.zarr`
  - `closures/brine.zarr`
- Onlap:
  - `onlap_segments_*.zarr`
  - `depth_maps_onlaps.zarr`
- Salt:
  - `salt_[0-9]*.zarr`
- Lithology:
  - `geology/faulted_lithology.zarr`
  - `faulted_lithology_*.zarr`
- Age/depth:
  - `geology/geologic_age.zarr`
  - `geologic_age_*.zarr`
  - `faulted_age_*.zarr`
  - `depth_maps.zarr`
  - `faulted_depth_*.zarr`

Important distinction:

- Global object image: one mask for the whole object type, such as `fault/fault`.
- Local object image: one mask for a graph object, such as `fault/fault_0` or `closure/closure_0`.

Closure works well because graph nodes map directly to fluid masks and DB extents.

Fault individual extraction depends on wrapper-generated `faults/fault_*.zarr`. Existing global `fault_segments` cannot reliably be split into individual faults after the build.

### `scripts/properties_2d_graph.py`

Copies DB-grounded properties graphs into view-specific 2D graphs.

It reads:

- `graphs/properties_graph`
- `build_objects/images/<sample_id>/*_object_position.json`

It writes:

- `graphs/properties_2d_graph`

It only updates node positions for object IDs already in the original properties graph. It does not add `salt_0`, `onlap_0`, etc. unless those nodes already exist in `properties_graph`.

### `Tracer/tracer.py`

Traverses graph JSON and emits simple relation records.

Relation shape:

```json
{
  "trace_type": "property",
  "source": "fault_0",
  "edge": "throw",
  "target": 16.3,
  "relation": ["fault_0", "throw", 16.3]
}
```

For edges:

```json
{
  "trace_type": "edge",
  "source": "fault",
  "edge": "REALIZED",
  "target": "fault_0"
}
```

### `NaturalTransform/text_transform.py`

Turns graph relations into natural text.

Examples:

```text
Fault 1 has throw of about 16.3892.
Fault 1 sits near x=426 and y=87.
Fault 1 occupies the area from x=346 to 506 and y=75 to 99.
Closure 1 contains brine.
Closure 1 avoids salt.
Salt is present.
```

Fields intentionally skipped from text:

- file paths
- image paths
- mask paths
- overlay paths
- wrapper source
- original fault index
- voxel bookkeeping
- model/config internals

Position and bbox fields are kept because they ground the VLM image task.

### `Verifier/create_rag.py`

Turns text evidence into RAG documents.

The RAG documents contain:

- `page_content`: natural evidence sentence
- metadata:
  - `source`
  - `edge`
  - `target`
  - `relation`

The current graph retrieval is semantic/vector-based with graph metadata. It is useful, but it does not guarantee perfect multi-hop evidence gathering.

### `Verifier/llm_machine.py`

Uses local vLLM-compatible `ChatOpenAI`.

Current model:

```text
Qwen/Qwen2.5-1.5B-Instruct
```

Current endpoint:

```text
http://localhost:8000/v1
```

It has two main chains:

- `question_generation()`
- `answer_generation()`

Prompts are intentionally strict:

- one question
- one atomic answer
- use only evidence
- do not mention graph/metadata/database/synthetic generation
- avoid broad interpretation unless evidence supports it

### `Verifier/generator_pipeline.py`

Newer hybrid RAG workflow.

Reads:

```text
graphs/properties_2d_graph
```

Writes:

```text
Dataset/hybrid_verified_qa.jsonl
```

Basic flow:

1. Select graph files by view.
2. Convert graph to text evidence.
3. Build vector/RAG store.
4. Generate one question from all evidence.
5. Retrieve evidence for the question.
6. Generate candidate answers.
7. Retrieve evidence for the answer.
8. Check question/answer evidence overlap.
9. Verify answer against evidence with LongTracer.
10. Save rows.

## Current Data Boundaries

### What belongs in `properties_graph`

DB-grounded geological metadata:

- fault count
- visible fault nodes
- fault throw
- fault tilt
- fault shear zone width
- fault gouge percentile
- closure fluid
- closure DB extents
- closure intersection flags
- salt inserted
- onlap episode count
- fan episode count
- sand/lithology summary fields

### What belongs in `properties_2d_graph`

Same factual graph, but with view-local position fields:

- `x`
- `y`
- `x_min`
- `x_max`
- `y_min`
- `y_max`
- `view`

This graph is for VLM image grounding.

### What belongs in image metadata

Per-object visual grounding:

- `sample_id`
- `object_type`
- `object_id`
- `view`
- `image_path`
- `mask_path`
- `overlay_path`
- `bbox`
- `center`

### What should not become natural dataset evidence

These are useful for debugging or linking files, but should not be spoken as answer evidence:

- image paths
- mask paths
- overlay paths
- fixed slice index
- wrapper source
- original fault index
- voxel count bookkeeping
- generation/config internals
- “synthetic”, “graph”, “metadata”, “database”

## Object Behavior

### Fault

Graph object:

- `fault`
- `fault_0`
- `fault_1`

Images:

- global `fault/fault`
- local `fault/fault_n` only when wrapper-generated masks exist

Important:

`fault_voxel_count_list` may include zero entries. Zero entries are attempted faults that did not become visible. The graph currently skips those and reindexes visible faults.

### Closure

Graph object:

- `closure`
- `closure_0`
- `closure_1`

Images:

- `closures/oil.zarr`
- `closures/gas.zarr`
- `closures/brine.zarr`

Closure is the most reliable object for graph/image alignment because each closure has:

- fluid
- bbox/extents
- voxel count
- intersection flags

### Salt

Graph state currently:

- usually `salt_inserted` on the category node

Image state:

- may have `salt_*.zarr`

Important:

Unless `properties_graph` contains a `salt` or `salt_0` node, the 2D graph will not have salt object positions. The text can say “Salt is present”, but not “Salt 1 sits near x/y” unless a salt object node is added.

### Onlap

Graph state:

- `number_onlap_episodes`
- `onlaps_horizon_list`

Image state:

- `onlap_segments_*.zarr`
- `depth_maps_onlaps.zarr`

Important:

Onlap individual component IDs are image-derived unless true graph object nodes exist. For graph-truth consistency, treat onlap as global unless object-level onlap graph nodes are added.

### Lithology / Age / Depth

Images may exist:

- `faulted_lithology_*.zarr`
- `geologic_age_*.zarr`
- `faulted_age_*.zarr`
- `depth_maps.zarr`
- `faulted_depth_*.zarr`

Graph evidence is currently mostly summary-level:

- `sand_voxel_pct`
- `sand_layer_percent_a_posteriori`
- fan/onlap summary keys

Do not use these arrays to invent semantic relations yet. They are useful for future derived reasoning, such as fault cutting age ranges, but that should be a separate validated stage.

## Current Commands

Generate DB/properties graphs from successful builds:

```bash
python scripts/graph_generator.py
```

Generate 2D object images and object position JSON:

```bash
python scripts/images_generator.py
```

Generate 2D properties graphs:

```bash
python scripts/properties_2d_graph.py
```

Inspect all evidence from one graph:

```bash
python - <<'PY'
from pathlib import Path
from Tracer.tracer import EvidenceTracer
from NaturalTransform import TextTransform

graph = sorted(Path("graphs/properties_2d_graph").glob("*.json"))[0]
relations = EvidenceTracer(graph).structural_evidence()
for item in TextTransform().relations_to_evidence(relations):
    print(item["sentence"])
PY
```

Run hybrid RAG dataset generation:

```bash
python Verifier/generator_pipeline.py
```

## Current Integration Note

There are two dataset paths in the repo:

1. `Dataset/pipeline.py`
   - older watcher-style pipeline
   - still defaults to `traces/views_graph`
   - imports `LLM` and `NLI`
   - may not match the current `Verifier` workflow without cleanup

2. `Verifier/generator_pipeline.py`
   - newer hybrid RAG workflow
   - defaults to `graphs/properties_2d_graph`
   - uses `Verifier/create_rag.py`, `Verifier/llm_machine.py`, and LongTracer

The current practical path is the newer hybrid workflow. If the goal is one unified dataset pipeline, the next cleanup should make `Dataset/pipeline.py` call the hybrid workflow or replace it with the newer view-based logic.

## Why Some Evidence Looks Missing

If RAG evidence says only:

```text
The section shows one fault.
Fault 1 ...
Closure 1 ...
Salt is present.
```

then the selected graph likely contains only:

```text
category:...
fault
fault_0
closure
closure_0
```

`properties_2d_graph.py` does not add missing objects from images. It only updates matching graph nodes. So if there is no `salt_0` node, there will be no salt bbox sentence. If there is no `fault_1`, there will be no Fault 2 sentence.

This is intentional for now. It keeps the graph factual and prevents image-only artifacts from becoming geology facts.

## Recommended Next Direction

Keep the split:

- `properties_graph`: DB-grounded geological truth
- `properties_2d_graph`: DB-grounded truth with view-local x/y/bbox
- image folders: visual grounding
- text transform: natural evidence only
- RAG/LLM: question-answer generation
- verifier: check whether answer is supported by graph evidence

For dataset generation by view:

1. Choose a view first: `inline`, `crossline`, or `timeslice`.
2. Select the matching 2D graph from `graphs/properties_2d_graph`.
3. Select the matching image/mask/overlay paths from `build_objects/images/<sample_id>`.
4. Generate question and answer from that graph evidence.
5. Verify the answer against graph evidence.
6. Export a row with:
   - instruction/question
   - answer
   - view
   - image path
   - optional mask/overlay path
   - graph evidence trace
   - verification result

Do not make the answer say which view it came from unless the final VLM task explicitly needs view classification.

