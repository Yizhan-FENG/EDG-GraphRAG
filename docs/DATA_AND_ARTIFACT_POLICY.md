# Data and artifact policy

This repository is intentionally code-only.

## Never commit

- raw incident reports, manuals, operational tickets, or user documents;
- processed text chunks, embeddings, vector stores, graph exports, or labels;
- training, validation, or test JSONL records;
- model-generated case reports and model-proxy review packets;
- checkpoints, LoRA adapters, tokenizer copies, or model caches;
- manuscript text, references, tables, figures, or submission correspondence;
- credentials, local account names, machine paths, or environment dumps.

## Allowed

- source code;
- JSON/Pydantic contracts;
- ontology and label schemas without instance data;
- configuration templates containing no local absolute paths or credentials;
- synthetic in-memory unit-test fixtures;
- experiment commands, metric definitions, and artifact schemas;
- aggregate documentation that does not reproduce private results.

## Recommended private layout

Keep data and outputs outside the Git checkout:

```text
private-research/
  data/
  catalogs/
  adapters/
  experiment-runs/
public-code/
  src/
  scripts/
  config/
```

Pass private paths through CLI arguments, ignored local YAML, or environment
variables. Use immutable manifest hashes to connect private runs to a public
code commit without publishing case content.

## Pre-publication check

Before every push:

```bash
git status --short
git ls-files
git grep -n -I -E "(sk-[A-Za-z0-9_-]{8,}|api[_-]?key|password|secret)"
```

Also inspect large files and forbidden extensions:

```bash
git ls-files | grep -E "\.(pdf|docx|pptx|xlsx|safetensors|gguf|bin|pt|pth|ckpt)$"
```

