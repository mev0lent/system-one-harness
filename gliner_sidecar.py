# GLiNER2.5-Decide sidecar: reads one JSON request per line on stdin, writes one JSON answer per line.
# Request:  {"id": ..., "state": "<serialised state>", "questions": {name: {type, instructions, criteria?}}}
# Answer:   {"id": ..., "answers": {name: {"type": "choice", "probabilities": {...}} | {"type": "noul", "noul": p}}, "ms": ...}
# Every question becomes one classification task with label descriptions and the instructions as its prompt.
# multi_label + cls_threshold 0 + softmax makes the library return the full distribution, not only the top label.
import json, sys, time
from gliner2 import AutoExtractor

REVISION = "9d1bfb848cd16d93ee56ccbc59cd92bf0f284d90"
# same option texts Laya uses for a Noul question
NOUL_LABELS = {"false": "no, the statement does not hold", "true": "yes, the statement holds"}

model = AutoExtractor.from_pretrained("fastino/GLiNER2.5-Decide", revision=REVISION)
print(json.dumps({"ready": True, "revision": REVISION}), flush=True)


def task(q):
    labels = q["criteria"] if q["type"] == "choice" else NOUL_LABELS
    return {"labels": labels, "multi_label": True, "cls_threshold": 0.0, "class_act": "softmax", "prompt": q["instructions"]}


for line in sys.stdin:
    req = json.loads(line)
    t = time.perf_counter()
    answers = {}
    # one call per question, mirroring the one-question-per-forward-pass setup used for Laya
    for name, q in req["questions"].items():
        out = model.classify_text(req["state"], {name: task(q)}, include_confidence=True)[name]
        probs = {o["label"]: o["confidence"] for o in out}
        answers[name] = {"type": "choice", "probabilities": probs} if q["type"] == "choice" else {"type": "noul", "noul": probs.get("true", 0.0)}
    print(json.dumps({"id": req["id"], "answers": answers, "ms": (time.perf_counter() - t) * 1000}), flush=True)
