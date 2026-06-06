import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from Tracer.tracer import EvidenceTracer
from NaturalTransform.text_transform import TextTransform


DEFAULT_NLI_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
DEFAULT_ENTAILMENT_THRESHOLD = 0.72
DEFAULT_CONTRADICTION_THRESHOLD = 0.72


def load_nli_model(model_name):
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise ImportError("NLI verification requires transformers and torch.") from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return tokenizer, model, torch


def label_index(model, target):
    id2label = {idx: label.lower() for idx, label in model.config.id2label.items()}
    for idx, label in id2label.items():
        if target in label:
            return idx
    raise ValueError(f"Model label mapping does not contain {target}: {model.config.id2label}")


class NliGraphVerifier:
    def __init__(
        self,
        model_name=DEFAULT_NLI_MODEL,
        top_k=8,
        entailment_threshold=DEFAULT_ENTAILMENT_THRESHOLD,
        contradiction_threshold=DEFAULT_CONTRADICTION_THRESHOLD,
    ):
        self.model_name = model_name
        self.top_k = top_k
        self.entailment_threshold = entailment_threshold
        self.contradiction_threshold = contradiction_threshold
        self.tokenizer, self.model, self.torch = load_nli_model(model_name)
        self.entailment_idx = label_index(self.model, "entail")
        self.contradiction_idx = label_index(self.model, "contrad")
        self.neutral_idx = label_index(self.model, "neutral")

    def verify_graph_claim(self, graph_path, claim):
        tracer = EvidenceTracer(graph_path)
        evidence = TextTransform().relations_to_evidence(tracer.retrieve(claim, top_k=self.top_k))
        scored = self.score_pairs(evidence, claim)
        best_entailment = max(scored, key=lambda item: item["scores"]["entailment"])
        best_contradiction = max(scored, key=lambda item: item["scores"]["contradiction"])

        if best_contradiction["scores"]["contradiction"] >= self.contradiction_threshold:
            status = "contradicted"
            deciding_evidence = best_contradiction
        elif best_entailment["scores"]["entailment"] >= self.entailment_threshold:
            status = "supported"
            deciding_evidence = best_entailment
        else:
            status = "insufficient_evidence"
            deciding_evidence = best_entailment

        return {
            "claim": claim,
            "status": status,
            "score": float(best_entailment["scores"]["entailment"]),
            "deciding_evidence": deciding_evidence,
            "retrieved_evidence": scored,
            "thresholds": {
                "entailment": self.entailment_threshold,
                "contradiction": self.contradiction_threshold,
            },
            "model": self.model_name,
        }

    def score_pairs(self, evidence_items, claim):
        premises = [item.get("sentence", item.get("text", "")) for item in evidence_items]
        hypotheses = [claim] * len(evidence_items)
        encoded = self.tokenizer(premises, hypotheses, padding=True, truncation=True, return_tensors="pt")

        with self.torch.no_grad():
            logits = self.model(**encoded).logits
            probabilities = self.torch.softmax(logits, dim=-1).cpu().numpy()

        scored = []
        for item, probs in zip(evidence_items, probabilities):
            scored.append({
                **item,
                "scores": {
                    "entailment": float(probs[self.entailment_idx]),
                    "contradiction": float(probs[self.contradiction_idx]),
                    "neutral": float(probs[self.neutral_idx]),
                },
            })
        return scored


def verify_claim_with_graph(graph_path, claim, model_name=DEFAULT_NLI_MODEL, top_k=8):
    verifier = NliGraphVerifier(model_name=model_name, top_k=top_k)
    return verifier.verify_graph_claim(graph_path, claim)
