# Synthetic Seismic Grounded-VQA and Referring-Segmentation Corpus — Dataset Documentation

*Source document for the dataset section of a CVPR-format paper. Written to be reimplementable and citable without access to the codebase. Numbers are exact and carry a provenance note; gaps are marked, not filled.*

> **Provenance scope (read first).** The shipped corpus is `Dataset/multimodal_multi_image_dataset.csv` (**1,467 rows**). Its intermediate artifacts (per-scene property graphs, the pre-CSV `verified_qa*.jsonl`) were deleted from disk before this document was written, and the on-disk generator/config/verifier are a **later revision** than the one that produced the corpus (evidence: on-disk `ratio_per_types` has only `full_mixed`/`boring` active with the 8-regime weights commented out; on-disk `QUESTION_PER_GRAPH=5` contradicts the corpus's max of 13 rows/scene; three verifier fixes on disk post-date the corpus, §14). Consequently: **corpus statistics (§11) are measured from the CSV and are exact; mechanism/thresholds (§5, §6, §9) are read from the current code and describe the current pipeline; several generation-time parameters specific to this corpus are `UNKNOWN — not recorded` and are marked as such.** Everything computed from the CSV was produced by `scripts/dataset_stats.py` (committed alongside this document); re-running it reproduces §11 exactly.

---

## 1. Scope

The corpus is a synthetic **seismic Grounded-VQA + referring-segmentation** dataset: each row pairs one or two 2D seismic section images with a natural-language question, a natural-language answer, one binary segmentation mask per referred object, and structured evidence that binds each answer fact to a ground-truth value. It targets a GLaMM/LISA-style model that emits a `<SEG>` token per localized object. It exists to supply **verified, physically-grounded supervision** for seismic interpretation—fault counting, dip/throw estimation, trap classification, spatial reasoning—where real labels are scarce and quantities such as fault throw are effectively unavailable from single-section interpretation. It is intended as **pretraining/augmentation**, not a substitute for real-data fine-tuning.

Totals *(scripts/dataset_stats.py over the CSV)*: **1,467 rows**, **197 distinct scenes**, **369 distinct images**, **369 (scene, view) pairs**, **1,729 total mask/region instances**. Scene coverage of the underlying build set is 197 of the 446 scene directories present in `build_objects/images/` (the balancing step, §12, dropped the rest).

---

## 2. Scene synthesis

**Forward model.** Synthoseis (open-source seismic forward simulator). Version: `UNKNOWN — not recorded in logs`. Citation: §16 (`UNVERIFIED`).

**Base generation parameters** *(settings.yaml, `control:` block; these are the base template—per-recipe regime settings override some, see below)*:

| parameter | value | note |
|---|---|---|
| `cube_shape` (volume) | **[100, 100, 500]** voxels (x, y, z) | small volume by design |
| `incident_angles` | **[7, 15, 24]°** | angle stack |
| `digi` (digitization) | **4** | |
| `infill_factor` | **2** | throw is stored as `throw / infill_factor × digi` → ms TWT (§5) |
| `bandwidth_ord` (wavelet order) | **4** | |
| `bandwidth_low` | **[3.0, 8.0] Hz** (range, sampled per scene) | |
| `bandwidth_high` | **[25.0, 45.0] Hz** (range) | wavelet passband |
| `signal_to_noise_ratio_db` | **[7.5, 12.5, 17.5] dB** | triangular [min, mode, max] sampled per scene |
| `initial_layer_stdev` | [3.0, 10.0] | |
| `thickness_min` / `thickness_max` | **4 / 18** voxels | layer thickness |
| `max_column_height` | [100.0, 150.0] | closure column |
| `min_closure_voxels_simple` / `_faulted` / `_onlap` | **500 / 1000 / 500** voxels | minimum closure size (§4) |
| `pad_samples` | 8 | |

**Where noise enters relative to labels.** Noise (Gaussian per the SNR range plus coherent migration artifacts) is added in the **amplitude/image** synthesis stage only; masks, counts, dip/throw/area, and all structural attributes are read from the **noise-free structural volumes and the parameter database**. Rationale: noise makes the *image* realistic without corrupting the *labels*, so a section can be arbitrarily noisy while its supervision stays exact. (Design statement, `scripts/config/settings.yaml` comment + `scripts/images/images_generator.py`; exact injection order not re-verified line-by-line per house-rule 4.)

**Structural regimes.** Eight regimes, each a recipe over Synthoseis parameters *(scripts/config/control_parameter.py, `SampleControl`)*:

| regime | fault count range | closure types | geological content |
|---|---|---|---|
| `boring` | 0 | simple | closures-only / featureless → negatives |
| `fault_only` | **1–9** | — | faults alone |
| `fault_complex` | **10–20** | simple | dense intersecting/branching faults + closures |
| `salt_only` | 0 | simple | salt + closures |
| `salt_fault_mixed` | **1–4** | simple | faults + salt + closures |
| `onlap` | 0 | **onlap** | onlap traps + closures |
| `depositional` | 0 | **simple + onlap** | closures + onlap |
| `full_mixed` | **2–6** | **simple + faulted + onlap** | everything (rarer) |

**Sampling weights.** `sample_population = 4` per recipe *(control_parameter.py)*, drawn repeatedly until the target scene count is reached; each sample's regime is drawn by weighted random choice from `ratio_per_types`. The **generation-time weights for this corpus are only partially recoverable**: the on-disk dict has `fault_complex 0.22, salt_fault_mixed 0.15, fault_only 0.13, onlap 0.12, salt_only 0.10, depositional 0.10` (commented out) and `full_mixed 0.5, boring 0.5` (active). The active `full_mixed/boring = 0.5` values are **inconsistent with the corpus** (which has full_mixed/boring at ~7% each), so the config on disk was edited after generation; the **exact generation-time weights for `full_mixed` and `boring` are `UNKNOWN — config edited post-generation`.** The **realized** per-regime scene distribution (which is what matters and is exact) is in §11. Rationale for weighting: over-weight geologically common fault/closure regimes for realism while keeping every regime represented for diversity (control_parameter.py comments).

---

## 3. Section extraction

2D sections are taken as **inline and crossline** slices from each 3D volume *(`view` field is `inline` or `crossline` in every region; scripts/images/images_generator.py, scripts/graph/properties_2d_graph.py)*. In the corpus, **369 distinct images across 197 scenes** → a mean of **1.87 images/scene** (369/197; not exactly 2 because balancing dropped some single-view rows). Slice **position sampling within the volume: `UNKNOWN — not recorded`** (graphs deleted; not in logs). Output image dimensions correspond to the crossline section of the `[100, 100, 500]` volume; exact pixel dimensions and amplitude normalization (percentile clip) are in `images_generator._normalize_image` (percentile [1, 99] clip, per code) — precise output resolution `UNKNOWN — not re-derived from a stored image header`. No resizing beyond the simulator's native section is recorded.

---

## 4. Classes and masks

Four object classes, with the numeric `class_id` used in the corpus *(scripts/dataset_stats.py; class_id read from the `regions` column)*:

| class | `class_id` | geological definition |
|---|---|---|
| fault | **1** | a slip surface; its 2D trace across the section |
| closure | **2** | a structural/stratigraphic trap (four-way, fault-dependent, or onlap), fluid-filled |
| salt | **3** | a salt body |
| onlap | **4** | an onlap termination surface / episode (aggregate per section) |

Masks are derived from the **noise-free structural volumes** (per object, not from the noisy image). **Minimum object-size thresholds** are enforced at the closure level via the simulator (`min_closure_voxels_simple = 500`, `_faulted = 1000`, `_onlap = 500` voxels, settings.yaml). A region/mask is emitted only if its mask has real pixels (design statement, DatasetMaker). The corpus contains **1,729 region/mask instances** total, of which **345 are section-level regions** ("the section", the whole-section union used for count/negative facts) and **1,384 are object regions** (§11).

---

## 5. Provenance table

One row per attribute the corpus emits. "Oracle" = whether the value is exact ground truth from the simulator database or measured from the mask pixels. *(Formulas/constants: scripts/graph/compute_attribute.py, scripts/graph/text_transform.py; value presence counts: scripts/dataset_stats.py.)*

| attribute | units | oracle | computation | why this oracle |
|---|---|---|---|---|
| object counts (`number_faults`, `number_hc_closures`, `number_fault_intersections`, `number_onlap_episodes`) | integer | **database** (per-view recount) | direct from the simulator model, recounted per 2D view for visibility | counts are a generation fact, not measurable robustly from one noisy section |
| fault **throw** | **ms TWT** | **database** | `throw_raw / infill_factor × digi` = `throw_raw / 2 × 4` (settings) | displacement is a generator parameter; from real data it requires cross-fault horizon correlation and is often unavailable |
| **fluid** | categorical {gas, oil, brine} | **database** | direct; oil/gas → "hydrocarbon-bearing", brine → "water-bearing" | fluid fill is set by the simulator; DHI/amplitude cues are indirect |
| **structural pattern** (`fault_mode`) | categorical {horst_and_graben, relay_ramp, branching, staircase} | **database** | direct; `random` emits no pattern sentence | a section-level generation setting; whitelisted to genuine geological patterns |
| object **intersections** (`intersects_fault`/`_salt`/`_onlap`) | boolean | **database** | direct simulator relation | trap dependence is set by the simulator's own closure-typing logic |
| apparent **dip** (`dip_deg`) | degrees, 0=flat…90=vertical | **mask** (RANSAC+PCA) | §6 | the model sees a 2D section; **apparent** dip is what is visually inferable (see below) |
| **area** coverage (`area_pct`) | % of section pixels, 1 dp | **mask** | `round(100.0 × mask.sum() / mask.size, 1)` | coverage is a visible geometric quantity, read from the same mask the model sees |
| **bounding box** | pixel [x_min,y_min,x_max,y_max] | **mask** | `argwhere(mask)` min/max over axes | read off the mask pixels; verified to reproduce them |
| **centroid** | pixel [x,y] | **mask** | bbox midpoint `[(x_min+x_max)/2, (y_min+y_max)/2]` | the coordinate name used to refer to the object |

**Apparent vs. true.** Dip is **apparent** (2D, from the mask), not true 3D dip. Rationale: true dip is under-determined from a single section, so a model that sees only that section cannot justify it; apparent dip is the honest, verifiable target for a single-view task. Throw, counts, fluid, pattern, and intersections are **exact 3D generation facts** projected to the view (counts are recounted per view for visibility). Value presence in the corpus *(scripts/dataset_stats.py)*: `area_pct` in 285 rows, `throw` in 134, `dip_deg` in 103; `number_faults` in 261, `fluid` in 102, `intersects_fault` in 75, `intersects_onlap` in 49, `number_onlap_episodes` in 47, `intersects_salt` in 39, `fault_mode` in 28, `number_fault_intersections` in 6, `number_hc_closures` in 6.

---

## 6. Geometric fitting detail (apparent dip)

Dip is the only pixel-fitted attribute *(scripts/graph/compute_attribute.py, `dip_degrees` / `ransac_inliers`)*. Procedure: on the fault mask's foreground pixels, **RANSAC** finds the largest set of collinear pixels, then **PCA** on those inliers gives the dominant line; dip = `degrees(atan2(|major_y|, |major_x|))`, rounded to 1 dp.

Exact thresholds and constants:

| constant | value | role |
|---|---|---|
| `_DIP_MIN_PIXELS` | **8** | mask needs ≥8 foreground px, else `None` |
| `_DIP_RANSAC_ITERS` | **300** | RANSAC iterations |
| `_DIP_RANSAC_THRESHOLD` | **1.5 px** | perpendicular distance for a pixel to count as on-line |
| `_DIP_MIN_INLIER_FRAC` | **0.5** | dominant line must contain ≥50% of the mask's pixels, else `None` |
| PCA eigenvalue-ratio gate | **> 0.6 → reject** | inliers too round (λ_min/λ_max > 0.6) → no dip |
| `_DIP_RANSAC_SEED` | **0** (fixed) | deterministic; graph generation reproducible |

**Fallback when the fit fails:** the attribute is simply omitted (returns `None`); no secondary estimator. Rationale: a contaminated fault mask (a crossing structure, a stray blob) would otherwise yield a geologically absurd near-horizontal dip, so it is better to emit no dip than a wrong one.

**Measured fit failure rate on this corpus:** the true per-fit failure rate is **`UNKNOWN — property graphs deleted; not recorded in logs`.** A recoverable proxy *(scripts/dataset_stats.py)*: of **331 fault object-regions** in the corpus, **103 carry a `dip_deg` value and 228 (68.9%) do not.** This 68.9% is an **upper bound conflated with question scope** — a fault region created to answer a throw or location question legitimately carries no dip even when the fit would have succeeded — so it overstates the true fit-failure rate by an unrecorded amount.

---

## 7. Evidence construction

Ground-truth facts become short, plain-text evidence sentences; the answer's grounding lives here, not in answer markup. A claim binds **object + attribute + value** by naming the object with its **coordinate** ("The fault at [84,377]") and stating the value in prose. Segmentation structure is one `<region>…<SEG>…</region>` block per mask, wrapped in `<evidence>`, index-aligned to the `regions` list (§8). Coordinates are the object's mask centroid and serve as the object's identity throughout (question, answer, evidence, and the coordinate entity-swap guard, §9).

Three verbatim evidence blocks from the corpus *(Dataset/multimodal_multi_image_dataset.csv)*:

1. `<evidence> <region> The section shows 3 faults. <SEG> </region> </evidence>`
2. `<evidence> <region> The fault at [84,377] has throw of about 53.64 ms. The fault at [84,377] dips at about 85.2 degrees. <SEG> </region> </evidence>`
3. `<evidence> <region> The fault at [99,83] has throw of about 219.77 ms. <SEG> </region> <region> The fault at [21.5,368] dips at about 81.2 degrees. <SEG> </region> </evidence>`

Note that evidence is **plain text with no value tags** — earlier revisions wrapped values in `<object>/<nums>/<center>/<bbox>`; the current pipeline emits plain prose (only the structural `<region>/<SEG>` remain), because the tags depressed retrieval/NLI similarity and were stripped before both stages anyway.

---

## 8. Question–answer generation

**Generator model.** `Qwen/Qwen2.5-1.5B-Instruct`, served locally by **sglang** at `--context-length 4096` *(Verifier/llm_machine.py, scripts/run_sglang.sh)*. Quantization: **`UNKNOWN — not recorded`** (served via `--mem-fraction-static 0.7`; weight dtype not logged).

**Decoding parameters** *(Verifier/llm_machine.py)*: two tuned client bindings on one model.
- **Question client:** temperature **0.6**, top_p **0.9**, frequency_penalty **0.6**, presence_penalty **1.2**.
- **Answer client:** temperature **0.1**, top_p **0.9** (answer binding), frequency_penalty **0.1**, presence_penalty **0.2**.
- Shared: `max_tokens = 640`, `n = 1`, retry `attempt = 5`; base defaults temp 0.2 / top_p 0.95.

**Prompt strategy.** A shared master prompt casts the model as a senior seismic interpreter, grounds every statement in the evidence only, forbids meta-terms, and requires numbers as digits. Question and answer prompts each enforce a strict JSON output contract and require a verbatim `RETRIEVAL_QUERY` (the exact evidence line(s) the item rests on) for verification lookup.

**Seed construction.** Object-level evidence docs are split into the section doc (`"<category> structure"`, carries counts/mode) and object docs; a **seed** = `OBJECTS_PER_SEED = 2` object docs + the section doc *(Verifier/generator_pipeline.py)*. A **typed seed** pairs same-type objects where possible so magnitude comparisons ("which dips more") are answerable (current code; this same-type pairing post-dates the corpus, §14). **Rows attempted per scene:** the loop targets `QUESTION_PER_GRAPH` passing rows, capped at `MAX_ATTEMPT = 3 × QUESTION_PER_GRAPH` question-batch attempts. On disk `QUESTION_PER_GRAPH = 5`; **the generation-time value for this corpus is `UNKNOWN` but was higher** — the corpus's max is **13 rows/scene** (§11), which `5` cannot produce, so the corpus was generated with a larger value (not recorded).

**Output structure** *(instruction field, verbatim)*: `"Answer the question with concise geological evidence. Reference specific objects using object tags. Insert one segmentation…"` The row carries: `evidence` (regions + `<SEG>`), `answer` (`<answer>…</answer>` plain prose), and a `regions` list with, per region: `image_idx, mask_idx, seg_idx, region_idx, object_type, object_name, view, class_id, bbox, center, values{measure, derive}`. The `<SEG>` count equals the region count in **1,467 / 1,467** rows (§11, alignment check).

**Two full verbatim rows** *(Dataset/multimodal_multi_image_dataset.csv)*:

*Single-object* (scene `seismic__0726_0442_full_mixed_…`):
- Q: `How many faults are shown in this section?`  ·  A: `<answer>There are 3 faults shown in this section.</answer>`
- evidence: `<evidence> <region> The section shows 3 faults. <SEG> </region> </evidence>`
- region[0]: `object_type=fault, object_name="the section", class_id=1, bbox=[73,249,99,505], center=[86.0,377.0], values={measure:{}, derive:{number_faults:3}}`

*Multi-object comparison* (scene `seismic__0726_0454_fault_complex_…`):
- Q: `Which fault is the steeper, and where does it end?`
- A: `<answer>The fault at [99,83] is steeper, dipping 65 degrees versus 45 for the fault at [21.5,368].</answer>`
- evidence: `<evidence> <region> The fault at [99,83] has throw of about 219.77 ms. <SEG> </region> <region> The fault at [21.5,368] dips at about 81.2 degrees. <SEG> </region> </evidence>`
- **Defect note (real, verbatim):** this answer's dip values (65, 45) appear **nowhere in its evidence** (which gives throw 219.77 ms and dip 81.2°) — the model **copied the few-shot example from the answer prompt verbatim**. It survived verification because the corpus predates the NLI label fix (§14); it is reported here rather than hidden.

---

## 9. Verification gates

Each candidate answer must pass a conjunction of gates *(Verifier/generator_pipeline.py, `best_answer` / `cover_answer`)*. **Rejection counts below are summed over the 5 shard journals (`journalctl --user -u seismic-qa-0..4`) and are CUMULATIVE across the full multi-restart generation history — they are NOT a clean single-run funnel for the 1,467-row corpus (§10).**

| # | gate | model / algorithm | threshold(s) | rejects | cumulative count | provably cannot catch |
|---|---|---|---|---|---|---|
| 0 | question retrieval | MiniLM `all-MiniLM-L6-v2` cosine over evidence | `MIN_RETRIEVAL_SCORE = 0.7` | question with no retrievable evidence | **5,034** | — |
| 1 | coverage (NLI) | hybrid: `all-MiniLM-L6-v2` STS gating cross-encoder `nli-deberta-v3-xsmall` | pass if `avg_sim ≥ 0.40` **or** `entailment > 0.5`, and **not** `contradiction > 0.5`; pipeline floor `MIN_ANSWER_TRUST = 0.80`; entailment floor `_MIN_ENTAILMENT = 0.5` (current) | answer fact not entailed by any evidence fact | **1,719** | a claim whose declared `RETRIEVAL_QUERY` is a *true but different* fact (answer text is not itself verified) |
| 2 | coordinate entity-swap | exact coordinate-set membership | every cited coord ∈ evidence coords | answer naming an object absent from evidence | **1,436** | a correct coord with a wrong value (that is gate 1's job) |
| 3 | question-coverage edge gate | edge-set subset | `q_edges ⊆ a_edges` | answer addressing a different attribute than asked | **6,262** | anything when `q_edges` is empty (skipped) |
| 4 | topic guard | object-type set disjointness | reject if question-types ∩ answer-types = ∅ | cross-type answer ("how many faults?"→"salt is present") | **112** | same-type-but-off-topic answers |
| 5 | count guard | `number_*` edge presence | count question must ground the matching `number_*` edge | count question answered without a count | **669** | a count grounded with the wrong integer of the right edge |
| 6 | attribute-consistency guard | claim-regex → required edge | a dip/throw/coverage claim must ground `dip_deg`/`throw`/`area_pct` | attribute asserted with no matching edge | **194** | a same-attribute wrong value (gate 1 contradiction) |
| — | `[ACCEPT]` (passed question stage) | — | — | — | **10,432** | — |

**Total rejections logged: 15,426** (all gates, cumulative). **Design rationale (measured, §5.3-style):** gates 4–6 are deterministic because the NLI (gate 1) verifies *truth*, not *relevance*, and—tested on the current pipeline—cannot distinguish a **grounded inference** ("A is steeper than B, 65 vs 45") from a **hallucination**: a whole-answer entailment check rejects the inference (entailment ≈ 0.004, and even contradiction ≈ 0.505 against its own facts) while passing the hallucination on shared-subject similarity, so structural checks are necessary. The core rejecters are the **edge gate (6,262)** and **question-retrieval (5,034)**; the added structural guards (topic 112 + count 669 + attribute 194 = **975**) are a small fraction.

---

## 10. Yield accounting

**Cumulative funnel (all shard journals, all restarts):** questions accepted at the retrieval stage `[ACCEPT] = 10,432` → answer-side rejections total `15,426` across gates 1–6 → **1,467 rows shipped**. **A clean per-run funnel (proposals → each gate → accepted) for the single run that produced the corpus is `UNKNOWN — the journals span many restarts and truncations; per-run boundaries not recorded`.** Per-graph the loop logged `[TALLY]` on **473** graphs (e.g. `attempts=15/15 passed=4`, `attempts=8/15 passed=5`).

**Wall-clock and hardware.** Hardware *(nproc / free -g / nvidia-smi, this machine)*: **12 CPU cores, 15 GB RAM, one 6 GB GPU**. Generation throughput *measured this session on the plain-evidence pipeline*: **≈350 verified rows/hour at 5 shard workers** (5 × `seismic-qa-*`, NLI on CPU, sglang on GPU); the earlier tagged-evidence pipeline ran **≈157 rows/hour** (§14). **Total wall-clock for the specific run that produced the 1,467-row corpus is `UNKNOWN — not recorded` (the run was restarted repeatedly during development).**

---

## 11. Corpus statistics

All tables are exact, produced by **`scripts/dataset_stats.py`** over `Dataset/multimodal_multi_image_dataset.csv` (**N = 1,467 rows**).

**Totals:** 1,467 rows · 197 distinct scenes · 369 distinct images · 369 (scene, view) pairs · 1,729 region/mask instances (1,384 object + 345 section-level).

**Scenes per regime** (regime parsed from `sample_id`):

| regime | scenes | % of 197 |
|---|---|---|
| fault_complex | 47 | 23.9% |
| salt_fault_mixed | 32 | 16.2% |
| fault_only | 30 | 15.2% |
| depositional | 24 | 12.2% |
| onlap | 18 | 9.1% |
| salt_only | 17 | 8.6% |
| boring | 15 | 7.6% |
| full_mixed | 14 | 7.1% |

**Rows per regime:** fault_complex 361 (24.6%), salt_fault_mixed 259 (17.7%), fault_only 240 (16.4%), depositional 174 (11.9%), salt_only 130 (8.9%), boring 109 (7.4%), full_mixed 97 (6.6%), onlap 97 (6.6%).

**Regions per object class** (segmentation targets, excluding the 345 section-level regions):

| class | class_id | regions | % of 1,729 |
|---|---|---|---|
| fault | 1 | 625 | 36.1% |
| onlap | 4 | 598 | 34.6% |
| closure | 2 | 351 | 20.3% |
| salt | 3 | 155 | 9.0% |

**Rows per question type** (regex heuristic, `dataset_stats.q_type`): area/coverage 415 (28.3%), location 346 (23.6%), count 319 (21.7%), comparison/relation 132 (9.0%), orientation/dip 65 (4.4%), fluid 56 (3.8%), throw 55 (3.7%), other 52 (3.5%), presence/absence 22 (1.5%), structural pattern 5 (0.3%).

**Rows per answer style:** extractive **1,331 (90.7%)**, qualitative **73 (5.0%)**, comparative **63 (4.3%)**.

**Objects per row** (distinct non-section object regions): 0 → 342 (23.3%), 1 → 908 (61.9%), 2 → 199 (13.6%), 3 → 8, 4 → 4, 5 → 2, 6 → 2, 8 → 2.

**Rows per scene:** min 1, median 7.0, mean 7.45, max 13.

**Answer length (words, tags stripped):** min 1, p25 7, median 8, p75 11, max 77, mean 10.1.

**`<SEG>` ↔ region alignment:** 1,467 / 1,467 rows aligned (SEG count = region count).

---

## 12. Composition control

Mechanisms that shape the *mix* (as opposed to correctness):

- **Weighted regime sampling** (§2) shapes scene-type mix. Realized distribution in §11; configured weights partially `UNKNOWN` (post-generation config edit).
- **Typed seeding** — pairing same-type objects in a seed so magnitude comparisons are answerable. *Current code; post-dates the corpus, so its before/after is not measurable on this corpus.*
- **Class rebalancing** — an inverse-frequency down-sampler (`scripts/qa_shards.sh balance`) that drops the over-represented class's lowest-collateral rows to a target (default: cap the top class to the 2nd-highest). **Measured before → after** on the pre-CSV jsonl *(this session)*: onlap-grounded rows **864 → 512**; fault/closure/salt untouched (512/345/217); **1,819 → 1,467 rows** (352 dropped). This rebalance is what produced the shipped corpus's row count.
- **Onlap-as-object fix** — a construction bug in which the aggregate onlap was filed as section context and thus attached to *every* seed, inflating it to **44% of regions** (pre-fix, measured on the pre-CSV jsonl this session). The fix reclassifies onlap as a normal object. **This fix post-dates the corpus**, so the shipped corpus still shows the inflation (onlap = 34.6% of regions, §11).

---

## 13. Splits

**No train/validation/test split has been taken.** `NOT RUN` — no split file, seed, or contiguity constraint exists in the repo. Prescription for whoever creates one: split **by scene, never by row**, because rows off one scene are correlated (same section, rephrased facts). **Effective sample size:** for a **vision** metric it is the **197 distinct scenes / 369 distinct images**, not the 1,467 rows; for a **language** metric it is closer to the 1,467 rows but still inflated by ~7.45 correlated rows/scene.

---

## 14. Measured failures on the current pipeline

Failures measured on this pipeline that led to a measured change *(all measured this session; the corpus was built before these landed, so the shipped corpus does not reflect them — see §15)*:

| failure | before | after | fix |
|---|---|---|---|
| cross-type Q/A mismatch on multi-type seeds | ~25% of sampled rows | 0% (sample of 12→then 14) | topic + count guards |
| context overflow on richer evidence | HTTP-400 on ~⅔ of graphs → 0 rows | resolved | sglang context 2048 → 4096 |
| onlap over-representation | 44% of regions | (source fixed; not re-run) | reclassify aggregate onlap as an object, not a section doc |
| **NLI label-order bug** | entailment/neutral **transposed** → neutral claims read as entailed (e.g. "onlap dipping" vs area: reported entail 0.992, true 0.001) | corrected mapping → entail 0.001 / true paraphrase 0.99 | relabel `compute_nli_scores` output |
| generation throughput | ~157 rows/hr (tagged evidence) | ~350 rows/hr (plain evidence) | remove value tags → shorter prompts |

**Measured-but-unfixed limitation:** the multi-object comparison example in §8 shows an answer that **copied the prompt's few-shot values (65/45) instead of the evidence values (throw 219.77, dip 81.2)** and passed verification, because the corpus was generated **before** the NLI label fix and entailment gate. The rate of such prompt-echo / value-mismatch answers in the shipped corpus is **`UNKNOWN — not systematically measured`** (one confirmed instance).

---

## 15. Not yet done

- **The shipped corpus was NOT regenerated** with the current pipeline's fixes: it lacks the **NLI label fix**, the **entailment coverage gate** (`_MIN_ENTAILMENT`), the **onlap-as-object fix**, the **same-type seed pairing**, and the **multi-object comparison prompt rule**. All exist in code (`NOT RUN` on the corpus).
- **No data split** (§13). `NOT RUN`.
- **No baseline / no trained model results** in this document. `NOT RUN` here.
- **No inter-annotator or human validation** of answer correctness beyond the automatic gates. `NOT RUN`.
- **No real-field evaluation** — the sim-to-real gap is unmeasured. `NOT RUN`.
- **Composition is emergent, not targeted:** the shipped corpus is 90.7% extractive, 61.9% single-object, 9.0% genuine comparison; the source-side controls to shift this (typed seeding, comparison prompting) post-date the corpus.
- **Dip fit-failure rate** is not directly recorded (§6); property graphs were deleted.
- **Generation config for this exact corpus** (regime weights for full_mixed/boring, `QUESTION_PER_GRAPH`, per-run funnel, wall-clock) is `UNKNOWN` (§2, §8, §10).
- **Perimeter, principal orientation, aspect ratio, compactness, convexity, fault length/curvature, horizon dip/curvature, heave, object-distance** are specified as computable (Appendix A) but **not emitted** by the corpus.

---

## 16. Citations

*Verified against nothing external in this environment; all marked `UNVERIFIED` — confirm author/venue/year against the primary source before use.*

- **Synthoseis** — synthetic seismic forward-modeling tool used to generate scenes. `UNVERIFIED` — cite the project repository/URL actually used; no peer-reviewed paper confirmed.
- **Qwen2.5-1.5B-Instruct** — generator model. `UNVERIFIED` (Qwen2.5 technical report).
- **sentence-transformers `all-MiniLM-L6-v2`** — STS / retrieval embedder. `UNVERIFIED` (Sentence-BERT / MiniLM).
- **cross-encoder `nli-deberta-v3-xsmall`** — NLI verifier; underlying **DeBERTa-v3**. `UNVERIFIED` (He et al., DeBERTa / DeBERTaV3).
- **LISA**, **GLaMM** — referring-segmentation LMMs (task format). `UNVERIFIED`.
- **RefCOCO/+/g**, **VQAv2**, **GQA** — referring-segmentation / VQA benchmarks (positioning). `UNVERIFIED`.
- **MNLI / ANLI** — NLI corpora behind the verifier. `UNVERIFIED`.
- **Real-field seismic dataset(s):** **none used.** This corpus is fully synthetic; no real-field release is referenced, so no canonical citation applies. If a real-field comparison is added, cite that dataset's canonical release paper explicitly.

---

## Appendix A — Attribute computability: synthetic vs. real-field

*(Framing that supports the §5/§7 argument; the pipeline currently emits only the starred rows.)*

| Group | Attributes | Synthetic (Synthoseis) | Real-field | Emitted here |
|---|---|---|---|---|
| 1. Single-object mask geometry | area\*, perimeter, centroid\*, bbox\*, orientation, aspect ratio, compactness, convexity | **exact** — mask ops on error-free masks + known Δx,Δz | computable; quality = interpretation quality | area, centroid, bbox |
| 2. Object curve geometry | fault length, curvature | **exact** — skeleton, or the generator surface | noisy from sparse sticks | — |
| 3. Line orientation | fault **dip**\*, horizon dip, curvature | **both** apparent (mask) and true (generator) | 2D → apparent only; true needs 3D | apparent `dip_deg` |
| 4. Fault–horizon | **throw**\*, heave, object distance, **intersection**\* | **exact — displacement from the generator** | reconstruct by cross-fault correlation; often unavailable | throw, intersections |
| 5. System statistics | **count**\*, density, spacing, mean dip, max throw | **exact** — complete model | only as complete as the interpretation | counts (per-view recounted) |

**Reading:** Group 4 (throw/heave) is the strongest synthetic-only advantage — exact displacement labels real single-section interpretation cannot cheaply supply. Groups 1–2 are a *cost* advantage (identical math on real masks), not a method advantage. Apparent-vs-true dip (Group 3) is a property of the single-view *task*, not a limitation of the data.
