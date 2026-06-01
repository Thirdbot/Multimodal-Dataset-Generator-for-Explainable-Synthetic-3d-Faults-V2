import json
import re
import sys
from pathlib import Path
from urllib import request
from urllib.error import HTTPError, URLError

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from Tracer.tracer import EvidenceTracer


DEFAULT_VLLM_ENDPOINT = "http://localhost:8000/v1/chat/completions"


def instruction_block(task="structural_interpretation"):
    if task == "fault_detection":
        return "Generate fault detection hypotheses grounded in the synthetic seismic sample evidence."
    return "Generate explanatory structural seismic interpretation hypotheses grounded in the synthetic sample evidence."


def is_fault_evidence(item):
    fact_name = str(item.get("fact_name", "")).lower()
    sentence = str(item.get("sentence", "")).lower()
    if "fault" in fact_name or "fault" in sentence:
        return True
    if fact_name == "has_category" and any(token in sentence for token in ("boring", "fault_only", "fault_complex")):
        return True
    return False


def select_prompt_evidence(tracer, evidence_limit, task="structural_interpretation"):
    source_evidence = tracer.structural_evidence()
    if task == "fault_detection":
        source_evidence = [item for item in source_evidence if is_fault_evidence(item)]

    priority = {"relation": 0, "property": 1, "sample": 2}
    selected = sorted(
        source_evidence,
        key=lambda item: (
            priority.get(item.get("source"), 9),
            str(item.get("fact_name", "")) == "HAS_CATEGORY",
        ),
    )
    return selected[:evidence_limit]


def build_hypothesis_prompt(graph_path, evidence_limit=12, hypothesis_count=5, task="structural_interpretation"):
    tracer = EvidenceTracer(graph_path)
    evidence = select_prompt_evidence(tracer, evidence_limit, task=task)
    evidence_lines = "\n".join(
        f"- {item['fact_name']}: {item['sentence']}"
        for item in evidence
    ) or "- No fault evidence was found in the graph trace."
    graph = tracer.graph
    sample_nodes = [node for node in graph.get("nodes", []) if node.get("label") == "Sample"]
    sample_id = sample_nodes[0].get("sample_id", graph_path.stem) if sample_nodes else Path(graph_path).stem

    task_rules = ""
    if task == "fault_detection":
        task_rules = """- Focus only on whether faults are present and how they affect visible seismic structure.
- Do not discuss closures, fluids, hydrocarbons, salt, or traps.
- If the evidence states zero faults or no fault evidence, produce no-fault hypotheses.
"""

    prompt = f"""Task: {instruction_block(task)}

Rules:
- Use only the evidence below.
- Do not explain your reasoning.
- Do not write analysis, notes, markdown, bullets, numbering, or prefaces.
- Return exactly {hypothesis_count} lines.
- Each line must start with "H: ".
- Each line must contain one standalone hypothesis.
- Do not copy evidence as raw comma-separated key-value pairs.
- Do not repeat the same hypothesis with different wording.
- Write as a natural seismic interpretation statement that could later be used to supervise a vision-language model.
- Mention how the observed properties combine into an interpretable structural scene.
- Prefer directly observable structural wording such as shows, contains, has, includes, or is characterized by.
- Each hypothesis may include a short evidence phrase using because, due to, or indicated by.
- Prefer qualitative summaries over exact coordinates or parameter values.
- If the evidence contains small counts, write them as words such as one, two, or three instead of digits.
- If the evidence states a fault count or closure count, prefer that explicit count in words instead of vague phrases like multiple, several, or many.
- Avoid exact x/y/z coordinates, voxel totals, throw values, and other dense numeric details unless there is no other way to make a correct statement.
- Do not mention graph, metadata, database, model parameters, or file paths.
- Do not repeat or paraphrase the task instructions.
- Do not talk about evidence lists, arrays, prompts, or what you are trying to do.
- Do not claim geological causality unless the evidence directly states causality.
- Prefer claims that NLI can verify directly from one or two evidence sentences.
- Each hypothesis should be one sentence of 18 to 45 words.
- Stop after the final hypothesis line.
- Never describe the output format.
{task_rules}

Sample id: {sample_id}

Evidence:
{evidence_lines}

Output:"""
    return prompt, evidence


def generate_with_vllm_endpoint(
    prompt,
    endpoint=DEFAULT_VLLM_ENDPOINT,
    model=None,
    api_key=None,
    max_new_tokens=2048,
    temperature=0.7,
    top_p=0.9,
    timeout=120,
    seed=42,
):
    payload = {
        "messages": [
            {"role": "system", "content": "Return only final hypotheses. Do not include reasoning."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    if model:
        payload["model"] = model
    if seed is not None:
        payload["seed"] = seed
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    encoded = json.dumps(payload).encode("utf-8")
    http_request = request.Request(endpoint, data=encoded, headers=headers, method="POST")
    try:
        with request.urlopen(http_request, timeout=timeout) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"vLLM endpoint returned HTTP {exc.code}: {body}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach vLLM endpoint at {endpoint}: {exc}") from exc

    choice = response_payload["choices"][0]
    text = choice["message"]["content"] if "message" in choice else choice["text"]
    return remove_reasoning_text(text).strip()


def remove_reasoning_text(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"^.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text


def clean_hypothesis_text(text):
    text = re.sub(r"\s+", " ", text).strip()
    match = re.search(r"(.+?[.!?])(?:\s|$)", text)
    if match:
        text = match.group(1).strip()
    text = text.strip(" -*")
    text = re.sub(
        r"^(?:HAS_CATEGORY|HAS_FAULT|HAS_CLOSURE|HAS_FLUID|REALIZED|CATEGORY|FAULTSYSTEM|MODELPARAMETERS|FAULT|CLOSURE)\s*:\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return text.strip()


def parse_hypotheses(raw_text):
    hypotheses = []
    seen = set()
    cleaned_text = remove_reasoning_text(raw_text)
    marker_pattern = re.compile(r"(?:^|\n)\s*(?:[-*]|\d+[.)])?\s*\**\s*H\s*:\s*", re.IGNORECASE)
    matches = list(marker_pattern.finditer(cleaned_text))

    if not matches:
        return []

    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(cleaned_text)
        hypothesis = clean_hypothesis_text(cleaned_text[start:end])
        normalized = hypothesis.lower()
        if is_usable_hypothesis(hypothesis) and normalized not in seen:
            hypotheses.append(hypothesis)
            seen.add(normalized)
    return hypotheses


def is_usable_hypothesis(text):
    normalized = text.lower().strip()
    if len(normalized.split()) < 6:
        return False
    if normalized[-1:] not in {".", "!", "?"}:
        return False

    banned_prefixes = (
        "has_category:",
        "has_fault:",
        "has_closure:",
        "has_fluid:",
        "faultsystem:",
        "category:",
        "modelparameters:",
        "closure:",
        "fault:",
        "realized:",
    )
    if normalized.startswith(banned_prefixes):
        return False

    banned_phrases = [
        "the task is",
        "the evidence provided",
        "i need to",
        "i'm trying to",
        "let me think",
        "looking at the evidence",
        "first,",
        "alright,",
        "the graph",
        "modelparameters",
        "file path",
        "output format",
        "array with role",
        "the sample includes a visible",
        "has_category:",
        "has_fault:",
        "has_closure:",
        "next,",
        "then,",
        "another hypothesis",
        "i should also",
        "i should",
        "i notice",
        "for the next hypothesis",
        "given this",
        "another",
        "belongs to the",
        "sample category is",
    ]
    for phrase in banned_phrases:
        if phrase in normalized:
            return False

    if normalized.startswith((
        "okay",
        "alright",
        "first",
        "now",
        "so",
        "wait",
        "next",
        "then",
        "another",
        "looking at",
        "given this",
        "for the next hypothesis",
        "i should",
        "i notice",
    )):
        return False

    bad_endings = (
        "because",
        "due to",
        "indicating",
        "showing",
        "suggesting",
        "with",
        "including",
    )
    if any(normalized.endswith(f" {ending}.") or normalized == f"{ending}." for ending in bad_endings):
        return False

    return True


def generate_hypotheses_for_graph(
    graph_path,
    endpoint=DEFAULT_VLLM_ENDPOINT,
    model=None,
    api_key=None,
    evidence_limit=12,
    count=5,
    max_new_tokens=1024,
    temperature=0.7,
    top_p=0.9,
    timeout=120,
    seed=42,
    task="structural_interpretation",
):
    prompt, evidence = build_hypothesis_prompt(
        graph_path,
        evidence_limit=evidence_limit,
        hypothesis_count=count,
        task=task,
    )
    raw_output = generate_with_vllm_endpoint(
        prompt,
        endpoint=endpoint,
        model=model,
        api_key=api_key,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        timeout=timeout,
        seed=seed,
    )
    return {
        "graph_path": str(graph_path),
        "prompt": prompt,
        "evidence": evidence,
        "raw_output": raw_output,
        "hypotheses": parse_hypotheses(raw_output),
    }
