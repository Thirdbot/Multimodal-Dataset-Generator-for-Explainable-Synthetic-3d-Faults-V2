import os
os.environ["MPLBACKEND"] = "Agg"  # non-interactive, thread-safe; image gen runs in worker threads

import watchfiles
from functools import partial
import json
import signal
import sys
import time
import asyncio
from pathlib import Path
import yaml
from watchfiles import Change
import shutil

from  scripts.graph.graph_generator import trace_success_tracker
from scripts.images.images_generator import generate_images_for_graph
from scripts.graph.properties_2d_graph import main as generate_properties_2d_graphs
from Verifier.generator_pipeline import RagWorkflow, DEFAULT_OUTPUT
from Dataset.DatasetMaker import main as build_dataset_csv

# recipes created -> build_configs file created (watch)

recipes_path = Path(__file__).parent.parent.parent / "recipes"
build_configs_path = Path(__file__).parent.parent.parent / "build_configs"
build_path = Path(__file__).parent.parent.parent / "builds"
builds_fail_path = Path(__file__).parent.parent.parent / "builds" / "fail.yaml"
builds_success_path = Path(__file__).parent.parent.parent / "builds" / "success.yaml"
properties_graph_path = Path(__file__).parent.parent.parent / "graphs" / "properties_graph"
images_path = Path(__file__).parent.parent.parent / "build_objects" / "images"
properties_2d_graph_path = Path(__file__).parent.parent.parent / "graphs" / "properties_2d_graph"


_image_gen_concurrency = max(1, int(os.environ.get("IMAGE_GEN_CONCURRENCY", "2")))
_image_gen_semaphore = asyncio.Semaphore(_image_gen_concurrency)


_build_concurrency = max(1, int(os.environ.get("BUILD_CONCURRENCY", "1")))
_build_semaphore = asyncio.Semaphore(_build_concurrency)
_build_timeout = float(os.environ.get("BUILD_TIMEOUT", "900"))  # seconds; kill a hung build so it can't wedge the queue

# success.yaml is rewritten once per finished build, and each rewrite fired an
# UNBOUNDED to_thread(trace_success_tracker, ...). When 10+ builds finish together
# that floods the shared default thread pool (~cpu+4 workers) and starves the event
# loop -> the whole pipeline stalls even though RAM is fine. Serialize tracing so it
# can't saturate the pool (idempotent, so re-reading the whole list stays cheap).
_trace_semaphore = asyncio.Semaphore(max(1, int(os.environ.get("TRACE_CONCURRENCY", "1"))))
_nli_device = os.environ.get("NLI_DEVICE", "cpu").strip().lower()  # cpu | cuda | auto -- keep NLI off the training GPU
STOP_SIGNAL = object()

_project_root = Path(__file__).parent.parent.parent


async def _run_trace(objs):
    async with _trace_semaphore:
        await asyncio.to_thread(trace_success_tracker, objs)

async def _run_build(config_path, run_id):
    # Each build runs in its OWN subprocess. Synthoseis monkeypatches process-global
    # state (fault settings + per-fault mask output) and is CPU-bound, so in-process
    # it would leak patches into the watcher AND hold the asyncio loop's GIL, stalling
    # image/graph/QA stages. A subprocess isolates both and the wait() is truly async.
    # Concurrency is capped by BUILD_CONCURRENCY (default 1 == the old serial behavior).
    # >1 is safe: sample_generator serializes success.yaml/failed.yaml writes with an
    # fcntl.flock lock file that holds across separate build processes.
    async with _build_semaphore:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "scripts.build.sample_generator",
            "--config", str(config_path), "--run-id", str(run_id),
            cwd=str(_project_root),
        )
        try:
            returncode = await asyncio.wait_for(proc.wait(), timeout=_build_timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            # Synthoseis occasionally hangs on a parameter combo; killing it frees the
            # queue. The partial build (no parameters.db) is swept on the next start.
            print(f"[BUILD TIMEOUT] {run_id} killed after {_build_timeout:.0f}s")
            return
        if returncode != 0:
            # sample_generator already recorded the failure in failed.yaml; this is
            # just visibility. A partial build with no parameters.db is swept later.
            print(f"[BUILD SUBPROCESS] {run_id} exited with code {returncode}")

def _sample_id_from_graph(graph_path):
    return Path(graph_path).stem.removesuffix("_db_extract_properties_graph")

def _delete_build_for_sample(sample_id):
    for folder in build_path.glob(sample_id):
        print(f"removing extracted build: {folder}")
        shutil.rmtree(folder, ignore_errors=True)

def _sweep_partial_builds(min_idle_seconds=120):
    now = time.time()
    for folder in build_path.glob("seismic__*"):
        if (folder / "parameters.db").exists():
            continue
        try:
            last = max((child.stat().st_mtime for child in folder.iterdir()),
                       default=folder.stat().st_mtime)
        except OSError:
            continue
        if now - last < min_idle_seconds:
            continue                                # recently written -> likely still building
        print(f"[SWEEP] removing interrupted build (no parameters.db): {folder.name}")
        shutil.rmtree(folder, ignore_errors=True)

async def _run_image_gen(graph_path):
    async with _image_gen_semaphore:
        graph_path = Path(graph_path)
        sample_id = _sample_id_from_graph(graph_path)
        scene_position = images_path / sample_id / "scene_position.json"
        if not scene_position.exists():  # skip re-imaging a sample already done
            try:
                await asyncio.to_thread(generate_images_for_graph, graph_path)
            except Exception as exc:
                print(f"[IMAGE GEN FAILED] {sample_id}: {exc}")
                return
        try:
            await asyncio.to_thread(generate_properties_2d_graphs, {sample_id})
        except Exception as exc:
            print(f"[2D GRAPH FAILED] {sample_id}: {exc}")
            return
        _delete_build_for_sample(sample_id)

async def on_recipes_delete(files,dest,types=""):
    for f_path in files:
        file_path = Path(dest).joinpath(f_path)
        print(f"removing file: {file_path}{types}")
        os.remove(f"{file_path}{types}")

async def on_build_delete(files,dest):
    file_path = Path(dest).glob(f"seismic__*_{files}")
    for f_path in file_path:
        print(f"removing file: {f_path}")
        shutil.rmtree(f_path)

async def on_build_failed(files):
    for f_path in files:
        if Path(f_path).exists:
            shutil.rmtree(f_path)
        else:
            continue


async def read_json(file_path):
    with open(file_path) as json_file:
        data = json.load(json_file)
        return data

async def read_yaml(file_path):
    with open(file_path) as yaml_file:
        data = yaml.safe_load(yaml_file)
        return data

async def recipes_watcher():
    recipes_dict = {}

    for p in recipes_path.glob("*.yaml"):
        recipes_dict[p.as_posix()] = await read_yaml(p)

    def data_callback(file_path,change):
        recipes_dict[file_path] = change.result()

    print("Watching recipes...")
    async for changes in watchfiles.awatch(recipes_path):
        for change_type, file_path in changes:
            if change_type in (Change.added,Change.modified):
                asyncio.create_task(read_yaml(file_path)).add_done_callback(partial(data_callback,file_path))
                print("Added/Modified recipes:",file_path)
            elif change_type == Change.deleted:
                data = recipes_dict.get(file_path)
                if not data:
                    print("No cached samples for deleted recipe:", file_path)
                    continue
                asyncio.create_task(on_recipes_delete(data['population']['samples'],build_configs_path.as_posix(),types=".json"))
                print("Deleted:",file_path)


def _config_already_built(run_id):
    if list(properties_graph_path.glob(f"seismic__*_{run_id}_db_extract_properties_graph.json")):
        return True
    if list(build_path.glob(f"seismic__*_{run_id}")):
        return True
    return False

async def builds_config_watcher():

    print("Watching builds configs...")

    for cfg in sorted(build_configs_path.glob("*.json")):
        if _config_already_built(cfg.stem):
            continue
    # what failed is failed, no rebuild from config

    async for changes in watchfiles.awatch(build_configs_path.as_posix()):
        for change_type, file_path in changes:
            if change_type == Change.added:
                print("Added configs:", file_path)
                asyncio.create_task(_run_build(file_path, Path(file_path).stem))
            elif change_type == Change.deleted:
                print("Deleted configs:", file_path)
                asyncio.create_task(on_build_delete(Path(file_path).stem,build_path))

async def builds_watcher():

    def read_add(change):
        try:
            data = change.result() or {}
        except Exception as exc:                                # torn read of success.yaml
            print(f"[SUCCESS READ FAILED] {exc}")
            return
        objs = [Path(p) for p in data.get('success_build_obj', [])]
        if not objs:
            return
        print("tracing success builds:", objs)
        asyncio.create_task(_run_trace(objs))

    def read_failed(change):
        try:
            data = change.result() or {}
        except Exception as exc:
            print(f"[FAILED READ FAILED] {exc}")
            return
        for cfg in data.get("failed_build_config", []):
            asyncio.create_task(on_build_delete(Path(cfg).stem, build_path))

    print("Watching success builds...")

    for folder in sorted(build_path.glob("seismic__*")):
        if not (folder / "parameters.db").exists():
            continue
        if (properties_graph_path / f"{folder.name}_db_extract_properties_graph.json").exists():
            continue
        print(f"[RECONCILE] tracing orphaned build: {folder.name}")
        asyncio.create_task(_run_trace([folder]))
    if builds_success_path.exists():                            # phase 1b: trace builds from success.yaml
        asyncio.create_task(read_yaml(builds_success_path.as_posix())).add_done_callback(read_add)
    async for changes in watchfiles.awatch(build_path.as_posix()):  # phase 2: new builds
        for change_type,file_path in changes:

            if change_type not in (Change.added, Change.modified):
                continue

            if Path(file_path).name == 'failed.yaml':
                print("failed.yaml changed:", file_path)
                # delete
                asyncio.create_task(read_yaml(file_path)).add_done_callback(read_failed)
            if Path(file_path).name == 'success.yaml':
                # pass to extract graph and graph properties building
                print("success.yaml changed:", file_path)
                asyncio.create_task(read_yaml(file_path)).add_done_callback(read_add)

async def graph_properties_watcher():
    print("Watching properties graph...")
    seen = set()
    for g in sorted(properties_graph_path.glob("*.json")):      # phase 1: existing graphs
        p = g.as_posix()
        seen.add(p)
        asyncio.create_task(_run_image_gen(p))
    async for changes in watchfiles.awatch(properties_graph_path.as_posix()):  # phase 2: new graphs
        for change_type, file_path in changes:
            if change_type not in (Change.added, Change.modified):
                continue
            p = Path(file_path).as_posix()
            if change_type == Change.added and p in seen:
                continue                                        # already handled in catch-up
            seen.add(p)
            asyncio.create_task(_run_image_gen(p))

async def dataset_gen_pipeline(queue):
    print("Watching 2d graphs -> enqueue dataset work...")
    seen = set()
    for g in sorted(properties_2d_graph_path.glob("*.json")):   # phase 1: existing files
        p = g.as_posix()
        seen.add(p)
        queue.put_nowait(p)
    async for changes in watchfiles.awatch(properties_2d_graph_path.as_posix()):  # phase 2: new files
        for change_type, file_path in changes:
            if change_type in (Change.added, Change.modified):
                p = Path(file_path).as_posix()
                if change_type == Change.added and p in seen:
                    continue                                    # already handled in catch-up
                seen.add(p)
                queue.put_nowait(p)

def _cuda_cleanup():
    """Release fragmented GPU cache so the next graph can retry after an OOM."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

def _init_nli_device():
    """Pin longtracer's NLI/STS verifier to NLI_DEVICE (default cpu) before its first
    use, so verification doesn't fight training/vLLM for the GPU. NLI_DEVICE=auto leaves
    sentence-transformers to choose (GPU if available). Must run before the first
    check() builds the shared model -- it early-returns once that model exists."""
    if _nli_device == "auto":
        print("[NLI] device=auto (sentence-transformers default, GPU if available)")
        return
    from longtracer.guard import nli_model as nm
    if nm._shared_model is not None:
        return
    orig_st, orig_ce = nm.SentenceTransformer, nm.CrossEncoder
    nm.SentenceTransformer = lambda *a, **k: orig_st(*a, **{**k, "device": _nli_device})
    nm.CrossEncoder = lambda *a, **k: orig_ce(*a, **{**k, "device": _nli_device})
    try:
        nm._shared_model = nm.HybridVerificationModel(verbose=False)
        print(f"[NLI] verifier pinned to device={_nli_device}")
    finally:
        nm.SentenceTransformer, nm.CrossEncoder = orig_st, orig_ce

def _seed_processed_graphs():
    """Graphs already turned into rows. A graph counts as processed only if its file
    has not been modified since the row recorded it, so a rebuilt graph is reprocessed
    on the next run. Set DATASET_REGEN=1 to reprocess everything (after a prompt change)."""
    stored = {}
    if DEFAULT_OUTPUT.exists():
        for line in DEFAULT_OUTPUT.read_text().splitlines():
            if not line.strip():
                continue
            try:
                meta = json.loads(line).get("metadata", {})
            except json.JSONDecodeError:
                continue
            gp = meta.get("graph_path")
            if not gp:
                continue
            gp = Path(gp).as_posix()
            stored[gp] = max(stored.get(gp, 0.0), float(meta.get("graph_mtime", 0.0) or 0.0))

    processed = set()
    for gp, mtime in stored.items():
        path = Path(gp)
        if not path.exists():
            processed.add(gp)                  # graph gone; nothing to reprocess
            continue
        try:
            if path.stat().st_mtime <= mtime + 1.0:  # not modified since (1s tolerance)
                processed.add(gp)
        except OSError:
            processed.add(gp)
    return processed

async def dataset_worker(queue):
    """Single consumer: append rows per new 2d-graph, serial, never truncating."""
    print("Dataset worker starting...")
    await asyncio.to_thread(_init_nli_device)   # pin NLI to NLI_DEVICE before the first check() (model load off-loop)
    workflow = None
    while workflow is None:                # keep retrying init so a transient OOM can't kill the worker
        try:
            workflow = await asyncio.to_thread(RagWorkflow)  # expensive init (embeddings + LLM)
        except Exception as exc:
            print(f"[DATASET] init failed, retrying: {exc}")
            _cuda_cleanup()
            await asyncio.sleep(5)
    regen = os.environ.get("DATASET_REGEN") == "1"   # force full rebuild after a prompt change
    workflow.start_output(truncate=regen)            # else append mode; keeps existing verified_qa.jsonl
    processed = set() if regen else _seed_processed_graphs()
    print("Dataset worker ready.")

    while True:
        first = await queue.get()          # blocks until data is ready
        if first is STOP_SIGNAL:
            queue.task_done()
            break

        batch = [first]
        stop = False
        while True:                        # coalesce whatever else is queued right now
            try:
                item = queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if item is STOP_SIGNAL:
                stop = True
                queue.task_done()          # account for the STOP item itself
                break
            batch.append(item)

        try:
            new_graphs = [g for g in dict.fromkeys(batch)
                          if g not in processed and Path(g).exists()]
            filtered_graph = [ Path(graph) for graph in new_graphs for view in ('inline','crossline') if view in Path(graph).name]

            made_rows = False
            for gp in filtered_graph:
                try:
                    print(f"[DATASET] generating rows for {Path(gp).name}")
                    await asyncio.to_thread(workflow.generate_for_graph, Path(gp))
                    processed.add(gp)      # mark done only on success
                    made_rows = True
                except Exception as exc:
                    # skip this graph, keep the worker alive; leave it unprocessed to retry
                    print(f"[DATASET] skip {Path(gp).name}: {exc}")
                    _cuda_cleanup()
            if made_rows:
                await asyncio.to_thread(build_dataset_csv)   # rebuild CSV from full jsonl
        finally:
            for _ in batch:                # exactly one task_done per non-STOP item pulled
                queue.task_done()

        if stop:
            break

# concurrent
async def gather():
    for p in (recipes_path, build_configs_path, build_path, properties_graph_path, images_path, properties_2d_graph_path):
        Path(p).mkdir(parents=True, exist_ok=True)
    _sweep_partial_builds()                          # clear interrupted builds before starting
    dataset_queue = asyncio.Queue()
    try:
        await asyncio.gather(
            recipes_watcher(),
            builds_config_watcher(),
            builds_watcher(),
            graph_properties_watcher(),
            dataset_gen_pipeline(dataset_queue),
            dataset_worker(dataset_queue),
            return_exceptions=False
            )
    except asyncio.CancelledError:
        await dataset_queue.put(STOP_SIGNAL)
        raise

def _install_stop_handlers(loop, main_task):
    # Low-level signal.signal handlers (not loop.add_signal_handler): the handler
    # body runs in the main thread and can os._exit *directly*, so stopping does
    # not depend on the event loop being free to process a cancellation -- a
    # Synthoseis build hogging the default thread pool can delay that. First
    # Ctrl+C tries a graceful cancel; a second one force-exits immediately.
    stopping = {"flag": False}

    def _handle(signum, frame):
        if stopping["flag"]:
            print("\n[Watcher] Second signal: forcing exit.", flush=True)
            os._exit(130)
        stopping["flag"] = True
        print("\n[Watcher] Stop signal received. Halting (press Ctrl+C again to force)...", flush=True)
        loop.call_soon_threadsafe(main_task.cancel)

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    main_task = loop.create_task(gather())
    _install_stop_handlers(loop, main_task)
    try:
        loop.run_until_complete(main_task)
    except asyncio.CancelledError:
        print("[Watcher] Watchers cancelled.")
    finally:
        # os._exit so we never block on shutdown_default_executor waiting for an
        # abandoned build thread; partial builds are cleared by _sweep_partial_builds
        # on the next start.
        print("[Watcher] Process terminated.", flush=True)
        sys.stdout.flush()
        os._exit(0)
