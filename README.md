# EDG-GraphRAG

Evidence-constrained multi-agent GraphRAG for auditable technical incident analysis.

EDG-GraphRAG separates extraction, deterministic knowledge-quality control,
evidence-grounded report generation, model-proxy review, and workflow routing
into four agents (A1--A4) plus a non-LLM state machine.

> Repository scope: code, configuration templates, ontology contracts, tests,
> and reproducible experiment procedures. Research data, trained adapters,
> model weights, case-level outputs, and manuscript files are intentionally
> excluded.

## Architecture

| Component | Implementation | Responsibility |
|---|---|---|
| A1 | Qwen3-4B-compatible local backbone + role LoRA | Extract entity, attribute, and relation candidates |
| A2 | Deterministic Python rules and ontology tools | Validate schema, provenance, relation endpoints, and knowledge tier |
| A3 | Shared local backbone + independent role LoRA | Produce a canonical evidence-bound diagnosis/report object |
| A4 | Local heterogeneous reviewer, e.g. GLM-4-9B 4-bit | Approve, revise, or reject a bounded report packet |
| Orchestrator | Non-LLM state machine | Bind artifacts, enforce transitions, persist auditable decisions |

The role sandbox is applied before adapter activation. LoRA changes generation
behaviour but never grants additional read/write permissions. Post-generation
validators enforce JSON contracts and evidence-ID whitelists.

## What is included

- `src/agent_kg/`: A1--A4 agents, contracts, model routing, role-LoRA runtime,
  quality metrics, and orchestration.
- `scripts/`: training, evaluation, ablation, retrieval, robustness, and local
  A4 execution entry points.
- `config/`: safe configuration examples. Paths are relative or supplied by
  environment variables.
- `kg_extensions/ontology_v2/`: ontology and bilingual label contracts.
- `prompts/`: role prompts.
- `tests/`: deterministic unit and contract tests.
- `docs/`: installation, data-boundary, and experiment-protocol documentation.

## What is not included

- raw or processed research data;
- fine-tuning JSONL files or frozen evaluation cases;
- knowledge-graph node/edge exports;
- prompts containing case material;
- model checkpoints, LoRA adapters, tokenizer copies, or caches;
- experiment predictions, audit packets, logs, or per-case results;
- the manuscript, references, figures, slides, or Word/PDF files;
- API keys, `.env` files, usernames, or machine-specific absolute paths.

See [Data and artifact policy](docs/DATA_AND_ARTIFACT_POLICY.md).

## Installation

Python 3.10--3.13 is supported.

```bash
git clone https://github.com/Yizhan-FENG/EDG-GraphRAG.git
cd EDG-GraphRAG
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Linux/macOS:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

For local QLoRA training:

```bash
pip install -e ".[train]"
```

For A4 local review, install [Ollama](https://ollama.com/) separately and make
the selected reviewer model available:

```bash
ollama pull glm4:9b
```

Detailed setup is in [Installation](docs/INSTALLATION.md).

## Configure

```bash
cp config/agents.example.yaml config/agents.yaml
```

On Windows PowerShell:

```powershell
Copy-Item config\agents.example.yaml config\agents.yaml
```

Edit only local model paths, adapter paths, and output directories. Do not put
secrets into YAML. If an OpenAI-compatible endpoint is added, reference an
environment-variable name instead of writing a token in the repository.

## Validate the code-only release

```bash
pytest -q
```

The tests use synthetic in-memory fixtures. They do not require or reconstruct
the research dataset.

## Reproduce the experiment process

The repository publishes the experimental *procedure*, not the private
research corpus or the paper's result files.

1. Prepare your own JSONL records using the schema in
   [Experiment workflow](docs/EXPERIMENT_WORKFLOW.md).
2. Create a versioned training catalog outside Git tracking.
3. Train A1 and A3 independently with one frozen backbone:

   ```bash
   python scripts/train_role_lora.py --config config/training/a1.example.yaml
   python scripts/train_role_lora.py --config config/training/a3.example.yaml
   ```

4. Run role-level evaluation:

   ```bash
   python scripts/evaluate_role_lora.py --role a1 --catalog /path/to/catalog.json
   python scripts/evaluate_role_lora.py --role a3 --catalog /path/to/catalog.json
   ```

5. Run the deterministic A2 audit and reviewed-label evaluation:

   ```bash
   python scripts/evaluate_a2_quality.py \
     --audit /path/to/a2_audit.json \
     --gold /path/to/reviewed_decisions.jsonl
   ```

6. Run paired ablations and the evidence-isolated GraphRAG evaluation using the
   scripts listed in `docs/EXPERIMENT_WORKFLOW.md`.
7. Run A4 locally. A4 outputs are model-proxy reviews, not human-expert gold
   labels.

Every experiment should write an immutable manifest containing the code
version, configuration hash, seed, data-manifest hash, model identity, adapter
identity, and output paths.

## Scientific interpretation

Contract validity, provenance compliance, deterministic robustness, and
repeatability are not substitutes for human-expert diagnostic accuracy. Users
must report those boundaries explicitly and must not describe model-proxy
agreement as expert ground truth.

## Security and privacy

Never commit operational incident records, personally identifying information,
credentials, proprietary manuals, or restricted infrastructure details.
Report accidental exposure privately and rotate affected credentials before
creating an issue.

## 中文说明

本仓库只公开核心代码、安装方法、配置模板、测试和可复现实验流程。原始数据、
微调数据、知识图谱导出、模型权重、LoRA、逐案例日志、实验中间结果以及论文正文
和配图均未上传。A4 的本地模型审议属于“模型代理评审”，不能表述为人工专家金标准。

## License

Code is released under the [MIT License](LICENSE). Dataset and model licenses
are separate and remain the responsibility of their respective owners.
