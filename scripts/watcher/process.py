import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import watchfiles
from functools import partial
import json
import asyncio
from pathlib import Path
import yaml
from anyio import to_thread
from watchfiles import Change
import shutil

from scripts.build.sample_generator import BuildGenerator
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


async def builds_config_watcher():

    print("Watching builds configs...")
    generator = BuildGenerator()

    async for changes in watchfiles.awatch(build_configs_path.as_posix()):
        for change_type, file_path in changes:
            if change_type == Change.added:
                print("Added configs:", file_path)
                # pass to build function
                asyncio.create_task(
                    asyncio.to_thread(generator.build_sample, file_path, Path(file_path).stem)
                )
            elif change_type == Change.deleted:
                print("Deleted configs:", file_path)
                asyncio.create_task(on_build_delete(Path(file_path).stem,build_path))

async def builds_watcher():

    def read_add(change):
        objs = [Path(p) for p in (change.result() or {}).get('success_build_obj', [])]
        if not objs:
            return
        print("tracing success builds:", objs)
        asyncio.create_task(asyncio.to_thread(trace_success_tracker, objs))

    def read_failed(change):
            for cfg in (change.result() or {}).get("failed_build_config", []):
                asyncio.create_task(on_build_delete(Path(cfg).stem, build_path))

    print("Watching success builds...")
    async for changes in watchfiles.awatch(build_path.as_posix()):
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
    async for changes in watchfiles.awatch(properties_graph_path.as_posix()):
        for change_type, file_path in changes:
            if change_type not in (Change.added, Change.modified):
                continue
            # forward to image generators
            asyncio.create_task(
                asyncio.to_thread(generate_images_for_graph, file_path)
            )

async def images_watcher():
    print("Watching images...")
    async for changes in watchfiles.awatch(images_path.as_posix()):
        if not any(t in (Change.added, Change.modified) for t, _ in changes):
            continue
        # forward to properties 2d graph (one batch rebuild per change-set)
        asyncio.create_task(
            asyncio.to_thread(generate_properties_2d_graphs)
        )

async def dataset_gen_pipeline(queue):
    """Producer: enqueue each new/modified 2d-graph file for row generation."""
    print("Watching 2d graphs -> enqueue dataset work...")
    async for changes in watchfiles.awatch(properties_2d_graph_path.as_posix()):
        for change_type, file_path in changes:
            if change_type in (Change.added, Change.modified):
                queue.put_nowait(Path(file_path).as_posix())

def _cuda_cleanup():
    """Release fragmented GPU cache so the next graph can retry after an OOM."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

def _seed_processed_graphs():
    """Graph files already turned into rows, so we never regenerate/override them."""
    processed = set()
    if DEFAULT_OUTPUT.exists():
        for line in DEFAULT_OUTPUT.read_text().splitlines():
            if not line.strip():
                continue
            try:
                gp = json.loads(line).get("metadata", {}).get("graph_path")
            except json.JSONDecodeError:
                continue
            if gp:
                processed.add(Path(gp).as_posix())
    return processed

async def dataset_worker(queue):
    """Single consumer: append rows per new 2d-graph, serial, never truncating."""
    print("Dataset worker starting...")
    workflow = None
    while workflow is None:                # keep retrying init so a transient OOM can't kill the worker
        try:
            workflow = await asyncio.to_thread(RagWorkflow)  # expensive init (embeddings + LLM)
        except Exception as exc:
            print(f"[DATASET] init failed, retrying: {exc}")
            _cuda_cleanup()
            await asyncio.sleep(5)
    workflow.start_output(truncate=False)  # append mode; keeps existing verified_qa.jsonl
    processed = _seed_processed_graphs()
    print("Dataset worker ready.")

    while True:
        first = await queue.get()          # blocks until data is ready
        got = 1
        batch = {first}
        try:                               # coalesce whatever else is queued right now
            while True:
                batch.add(queue.get_nowait())
                got += 1
        except asyncio.QueueEmpty:
            pass

        try:
            new_graphs = [g for g in batch
                          if g not in processed and Path(g).exists()]
            made_rows = False
            for gp in new_graphs:
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
            for _ in range(got):
                queue.task_done()

# concurrent
async def gather():
    for p in (build_path, properties_graph_path, images_path, properties_2d_graph_path):
        Path(p).mkdir(parents=True, exist_ok=True)
    dataset_queue = asyncio.Queue()
    await asyncio.gather(
        recipes_watcher(),
        builds_config_watcher(),
        builds_watcher(),
        graph_properties_watcher(),
        images_watcher(),
        dataset_gen_pipeline(dataset_queue),
        dataset_worker(dataset_queue),
        return_exceptions=True
        )


if __name__ == "__main__":
    try:
        asyncio.run(gather())
    except KeyboardInterrupt:
        print("Interrupted!!!")