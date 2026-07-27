# Reproducible experiment workflow

This document publishes the experiment process without publishing research
records or manuscript results.

## 1. Record contract

Training records are JSONL objects with exactly three chat messages:

```json
{
  "messages": [
    {"role": "system", "content": "role contract"},
    {"role": "user", "content": "input packet with evidence identifiers"},
    {"role": "assistant", "content": "{\"structured\": \"target\"}"}
  ],
  "metadata": {
    "sample_id": "stable-private-id",
    "source_name": "private-source-tier",
    "split": "train"
  }
}
```

Do not place actual records in this repository.

## 2. Dataset catalog

The training runner expects a versioned catalog with `dataset_version: "0.3"`,
role sections (`a1`, `a3`), per-source train/validation/test paths and counts,
plus a `forbidden_inputs` list. Test records must never be selected by a
training configuration.

Store the catalog outside Git and hash it in each run manifest.

## 3. Training

Train A1 and A3 separately. The base model remains frozen and only one adapter
is active for a request.

```bash
python scripts/train_role_lora.py --config config/training/a1.example.yaml
python scripts/train_role_lora.py --config config/training/a3.example.yaml
```

Use `run_registered_training.py` for segmented recovery. It resumes only from
the latest registered checkpoint and appends to an orchestrator log.

## 4. Frozen evaluation

Freeze sample IDs before inspecting results. Record:

- code commit;
- catalog hash;
- adapter hash;
- model revision;
- decode parameters;
- random seed;
- environment and GPU metadata.

Role evaluation:

```bash
python scripts/evaluate_role_lora.py --role a1 --catalog /private/catalog.json
python scripts/evaluate_role_lora.py --role a3 --catalog /private/catalog.json
```

Report structural JSON validity separately from semantic entity/relation
metrics. A parser or orchestrator repair must not be presented as raw model
output.

## 5. A2 quality control

A2 is deterministic. Report the confirmed/candidate/rejected partition,
schema validity, evidence/provenance validity, endpoint constraints, self-loop
filtering, and policy version.

```bash
python scripts/evaluate_a2_quality.py \
  --audit /private/run/a2_audit.json \
  --gold /private/review/reviewed_decisions.jsonl
```

If no human-reviewed relation labels exist, do not report expert accuracy.
Use rule-contract robustness and explicitly label any model review as proxy
assessment.

## 6. A3 and GraphRAG

Evaluate canonical report JSON, section coverage, evidence precision/recall/F1,
and source-scope compliance. Cross-case retrieval cards are non-evidentiary
context and must not carry current-case citation authority.

Use paired case IDs for the no-retrieval and leave-one-case-out retrieval arms.

## 7. A4 local adjudication

A4 receives one bounded packet containing an A3 draft, A2 result, and current
case evidence whitelist. It returns only an approve/revise/reject contract.

Run A4 through localhost:

```bash
python scripts/run_a4_local_ollama_queue.py \
  --limit 5 \
  --label local_a4_registered_run
```

Persist decision, bounded issues, revision instructions, model identity,
latency, request fingerprint, and final state-machine routing. Do not persist
hidden chain of thought.

## 8. Ablation controls

Use `config/experiments/ablation_matrix.example.yaml`. GPU jobs must run
serially on a single-GPU machine. Compare paired records and retain negative
results.

## 9. Robustness

Register deterministic mutations before execution:

- missing evidence ID;
- cross-case evidence ID;
- self-loop relation;
- endpoint-type mismatch;
- low confidence;
- modifier loss;
- unsafe action request.

Expected outcomes must be defined by the public contract, not chosen after
seeing model output.

## 10. Reporting boundaries

Contract compliance, source-scope enforcement, and deterministic mutation
tests are software/interface evidence. They do not establish operational
diagnostic correctness, report usefulness, or human-expert agreement.

