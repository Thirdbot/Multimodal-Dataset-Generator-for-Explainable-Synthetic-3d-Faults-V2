"""One-shot QA runner (clean regen).

Runs generate_multimodal_dataset over QA_GRAPH_ROOT (default: the full graph dir),
truncating Dataset/verified_qa.jsonl once at the start and appending rows as it goes.
Point QA_GRAPH_ROOT at a scoped symlink dir (e.g. graphs/_new_only) to regen a subset.

The verifier (DeBERTa NLI + STS) MUST stay off the GPU -- that GPU is held by the sglang
Qwen server, and letting NLI grab CUDA OOMs it (CUBLAS_STATUS_ALLOC_FAILED). The watcher
does this via _init_nli_device(); replicated here since this bypasses the watcher.
"""
import os, json

os.environ.setdefault("NLI_DEVICE", "cpu")


def _pin_nli_cpu():
    """Mirror scripts.watcher.process._init_nli_device: force longtracer's shared
    verifier onto CPU before the first check() builds it."""
    from longtracer.guard import nli_model as nm
    if nm._shared_model is not None:
        return
    orig_st, orig_ce = nm.SentenceTransformer, nm.CrossEncoder
    nm.SentenceTransformer = lambda *a, **k: orig_st(*a, **{**k, "device": "cpu"})
    nm.CrossEncoder = lambda *a, **k: orig_ce(*a, **{**k, "device": "cpu"})
    try:
        nm._shared_model = nm.HybridVerificationModel(verbose=False)
        print("[NLI] verifier pinned to device=cpu")
    finally:
        nm.SentenceTransformer, nm.CrossEncoder = orig_st, orig_ce


_pin_nli_cpu()

from Verifier.generator_pipeline import generate_multimodal_dataset, DEFAULT_OUTPUT

graph_root = os.environ.get("QA_GRAPH_ROOT", "graphs/properties_2d_graph")
output_path = os.environ.get("QA_OUTPUT", str(DEFAULT_OUTPUT))
resume = os.environ.get("QA_RESUME", "").strip() in ("1", "true", "yes")  # append + skip done graphs
rows = generate_multimodal_dataset(graph_root=graph_root, output_path=output_path, resume=resume)
print(json.dumps({"rows": len(rows), "output": output_path, "graph_root": graph_root, "resume": resume}))
