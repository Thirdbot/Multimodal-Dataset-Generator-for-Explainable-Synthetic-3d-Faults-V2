"""Fix a label-order bug in longtracer's NLI reader.

The `cross-encoder/nli-deberta-v3-*` models output logits ordered
    [contradiction, entailment, neutral]   (per the model's own id2label)
but `longtracer.guard.nli_model.compute_nli_scores` maps them as
    [contradiction, neutral, entailment]   -- swapping entailment <-> neutral.

`contradiction` (index 0) is correct, which is why *value* errors were still caught, but every
`entailment`/`neutral` read was inverted -- so a NEUTRAL claim ("onlap is dipping", grounded on
an area fact) was reported as high entailment and passed, while a true paraphrase was reported
as neutral. This re-wraps `compute_nli_scores` to return the correct mapping, making the
entailment signal trustworthy so the verifier can require real entailment.

Call `apply_nli_label_fix()` once before the first `check()` (idempotent).
"""


def apply_nli_label_fix():
    from longtracer.guard import nli_model as nm
    HVM = nm.HybridVerificationModel
    if getattr(HVM, "_labels_fixed", False):
        return
    _orig = HVM.compute_nli_scores

    def compute_nli_scores(self, source, claim):
        d = _orig(self, source, claim)
        # _orig's "neutral" is really entailment; its "entailment" is really neutral -> swap back.
        return {
            "contradiction": d["contradiction"],
            "entailment": d["neutral"],
            "neutral": d["entailment"],
        }

    HVM.compute_nli_scores = compute_nli_scores
    HVM._labels_fixed = True
