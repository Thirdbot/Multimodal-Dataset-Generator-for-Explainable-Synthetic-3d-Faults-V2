import json
import re
import sys
from pathlib import Path
from urllib import request
from urllib.error import HTTPError, URLError

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from Tracer.tracer import EvidenceTracer
from NaturalTransform.text_transform import TextTransform


DEFAULT_VLLM_ENDPOINT = "http://localhost:8000/v1/chat/completions"

def select_prompt_evidence(tracer, evidence_limit=None, **_):
    source_evidence = TextTransform().relations_to_evidence(tracer.structural_evidence())
    return source_evidence if evidence_limit is None else source_evidence[:evidence_limit]


def build_hypothesis_prompt(graph_path, evidence_limit=None, hypothesis_count=5, question=None):
    tracer = EvidenceTracer(graph_path)
    evidence = select_prompt_evidence(tracer, evidence_limit)
    evidence_lines = "\n".join(
        f"- {item.get('sentence', '')}"
        for item in evidence
    ) or "- No evidence was found."
    graph = tracer.graph
    sample_nodes = [node for node in graph.get("nodes", []) if node.get("label") == "Sample"]
    sample_id = sample_nodes[0].get("sample_id", graph_path.stem) if sample_nodes else Path(graph_path).stem

    question_block = f"\nQuestion:\n{question}\n" if question else ""
    task_line = "Answer the question using the evidence." if question else "Generate natural interpretation hypotheses grounded in the evidence."

    prompt = f"""Task: {task_line}

Rules:
- Use only the evidence below.
- Do not mention graph, metadata, database, model parameters, file paths, or prompts.
- Return exactly {hypothesis_count} lines.
- Each line must start with "H: ".
- Write one standalone sentence per line.
- Keep the wording natural and concise.
- Combine related evidence when possible.
- Do not add facts that are not in the evidence.
- If a question is provided, answer that question directly.

Sample id: {sample_id}
{question_block}

Evidence:
{evidence_lines}

Output:"""
    return prompt, evidence


def build_question_prompt(graph_path, evidence_limit=None, question_count=5):
    tracer = EvidenceTracer(graph_path)
    evidence = select_prompt_evidence(tracer, evidence_limit)
    evidence_lines = "\n".join(
        f"- {item.get('sentence', '')}"
        for item in evidence
    ) or "- No evidence was found."
    sample_id = Path(graph_path).stem

    prompt = f"""Task: Generate natural questions that can be answered from the evidence.

Rules:
- Use only the evidence below.
- Return exactly {question_count} lines.
- Each line must start with "Q: ".
- Write one standalone question per line.
- Ask about visible or interpretable features supported by the evidence.
- Do not ask why something formed or what caused it unless the evidence states that.
- Do not mention graph, metadata, database, model parameters, file paths, or prompts.
- Do not answer the question.

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


def clean_question_text(text):
    text = re.sub(r"\s+", " ", text).strip()
    match = re.search(r"(.+?\\?)(?:\\s|$)", text)
    if match:
        text = match.group(1).strip()
    text = text.strip(" -*")
    return text


def parse_questions(raw_text):
    questions = []
    seen = set()
    cleaned_text = remove_reasoning_text(raw_text)
    marker_pattern = re.compile(r"(?:^|\n)\s*(?:[-*]|\d+[.)])?\s*\**\s*Q\s*:\s*", re.IGNORECASE)
    matches = list(marker_pattern.finditer(cleaned_text))
    if not matches:
        return []

    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(cleaned_text)
        question = clean_question_text(cleaned_text[start:end])
        normalized = question.lower()
        if is_usable_question(question) and normalized not in seen:
            questions.append(question)
            seen.add(normalized)
    return questions


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


def is_usable_question(text):
    normalized = text.lower().strip()
    if len(normalized.split()) < 4:
        return False
    if not normalized.endswith("?"):
        return False
    banned_phrases = [
        "the graph",
        "metadata",
        "database",
        "model parameter",
        "file path",
        "prompt",
        "why did",
        "what caused",
        "how was",
    ]
    return not any(phrase in normalized for phrase in banned_phrases)


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
    question=None,
):
    prompt, evidence = build_hypothesis_prompt(
        graph_path,
        evidence_limit=evidence_limit,
        hypothesis_count=count,
        question=question,
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


def generate_questions_for_graph(
    graph_path,
    endpoint=DEFAULT_VLLM_ENDPOINT,
    model=None,
    api_key=None,
    evidence_limit=None,
    count=5,
    max_new_tokens=1024,
    temperature=0.7,
    top_p=0.9,
    timeout=120,
    seed=42,
):
    prompt, evidence = build_question_prompt(
        graph_path,
        evidence_limit=evidence_limit,
        question_count=count,
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
        "questions": parse_questions(raw_output),
    }
