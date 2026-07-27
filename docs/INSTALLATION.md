# Installation

## 1. Requirements

- Python 3.10--3.13.
- Git.
- A CUDA-capable GPU for local QLoRA training.
- Ollama for the optional local A4 reviewer.
- Enough local storage for independently downloaded model weights.

The code-only tests run on CPU.

## 2. Create an isolated environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

## 3. Optional training dependencies

```bash
pip install -e ".[train]"
```

Install the CUDA-compatible PyTorch build recommended for your driver before
starting QLoRA. Do not assume that a generic wheel matches the local CUDA
runtime.

## 4. Configuration

Copy the safe template:

```powershell
Copy-Item config\agents.example.yaml config\agents.yaml
```

Set model and adapter paths with environment variables. Keep the real `.env`
outside Git.

## 5. Local A4 model

Install Ollama, then:

```bash
ollama pull glm4:9b
ollama list
```

The public default binds A4 to localhost. A4 audit outputs are model-proxy
decisions and must not be labelled as human-expert review.

## 6. Verify

```bash
pytest -q
```

For linting:

```bash
ruff check src tests scripts
```

