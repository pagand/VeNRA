#  VeNRA: Verifiable Numerical Reasoning Agent

> **"Trust, but Verify." — The core philosophy of Financial AI**

<p align="center">
  <img src="assets/structure.webp" alt="VeNRA Architecture" width="55%">
</p>

<p align="center">

<a href="https://arxiv.org/pdf/2603.04663">
  <img src="https://img.shields.io/badge/Paper-ArXiv-b31b1b?logo=arxiv&logoColor=white">
</a>

<a href="https://huggingface.co/datasets/pagand/venra">
  <img src="https://img.shields.io/badge/Dataset-HuggingFace-yellow?logo=huggingface&logoColor=black">
</a>

<a href="https://huggingface.co/spaces/pagand/VeNRA_halDet">
  <img src="https://img.shields.io/badge/Demo-HuggingFace%20Space-blue?logo=huggingface&logoColor=white">
</a>

<a href="https://huggingface.co/pagand/venra">
  <img src="https://img.shields.io/badge/Model-LoRA%20Adapter-green?logo=huggingface&logoColor=white">
</a>

<a href="https://github.com/pagand/VeNRA/stargazers">
 <img src="https://img.shields.io/github/stars/pagand/VeNRA?style=social">
</a>

</p>

---

## Overview

**VeNRA (Verifiable Numerical Reasoning Agent)** is a **neuro-symbolic financial reasoning system** designed to eliminate hallucinated numerical outputs from Large Language Models.

Traditional Retrieval-Augmented Generation (RAG) systems struggle in **deterministic domains like finance** because:

* LLMs are **probabilistic token predictors**, not arithmetic engines
* Dense retrieval introduces **semantic conflation** (e.g. *Net Income* vs *Net Sales*)
* Minor numerical errors destroy operational trust

VeNRA addresses these limitations through a **hybrid architecture that separates reasoning from computation** and introduces an **independent auditing layer**.

The system enables users to ask complex financial questions such as:

> *"How much the sale excluding acquisition increase compare to last year and what it was due to?"*

and receive answers that are:

* **Mathematically deterministic**
* **Fully traceable**
* **Audited in real time**

---

## Key Features

### 🧮 Deterministic Numerical Reasoning
All arithmetic is executed through **Python code generation and execution**, not token prediction.

### 🔎 Universal Fact Ledger (UFL)
Financial statements are parsed into a **typed variable ledger**, replacing probabilistic document retrieval.

### 🔗 Full Traceability
Every numeric answer is linked to:

* the exact **financial statement row**
* the **source document span**
* the **computation trace**

### 🛡 Sentinel Hallucination Detector
A lightweight **3B parameter forensic model** audits reasoning traces before answers are returned.

### ⚡ Hybrid Retrieval
Combines:

* lexical filtering (exact financial metrics)
* semantic search (concept discovery)

to prevent vector-space conflation.

---

## Architecture

VeNRA is built around three modular components:

### 1. Ingestion Engine
Parses financial PDFs into a structured **Universal Fact Ledger** with strict schema validation.

### 2. Runtime Agent
A **Program-Aided LLM agent** generates deterministic Python programs that compute answers from ledger variables.

### 3. Sentinel Service
A separate **forensic auditing model** that verifies reasoning traces and produces a groundedness score.

---

## Installation

### Prerequisites

* Python **3.11+**
* API keys for reasoning and monitoring services

---

### Clone Repository

```bash
git clone https://github.com/pagand/VeNRA.git
cd VeNRA
````

---

### Configure Environment

```bash
cp .env.example .env
```

Add your API credentials to `.env`.

---

## Environment Setup

Dependencies differ between **training** and **serving**.

You only need one depending on your goal.

---

# Training Environment

### Automatic Setup

```bash
chmod +x setup.sh
./setup.sh
source .venv/bin/activate
```

---

### Manual Setup

Create environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Upgrade tooling

```bash
pip install --upgrade pip setuptools wheel
```

Install PyTorch

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
--index-url https://download.pytorch.org/whl/cu124
```

Install low-level dependencies

```bash
pip install triton==3.1.0 bitsandbytes==0.43.3
```

Verify installation

```bash
python << 'EOF'
import torch, bitsandbytes as bnb, triton
print(f"✓ PyTorch {torch.__version__}")
print(f"✓ CUDA {torch.version.cuda}")
print(f"✓ triton {triton.__version__}")
print(f"✓ bitsandbytes {bnb.__version__}")
print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
EOF
```

Install remaining dependencies

```bash
pip install -r requirements_training.txt
```

---

# Serving Environment

Create environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements_serving.txt
```

Run the service

```bash
uvicorn venra.main:app --reload --app-dir src
```

The API will be available at

```
http://localhost:8000
```

---

## Training the Sentinel Model

Start training

```bash
python src/hal_det/training/train.py --output_dir ./data/outputs
```

Quick test run

```bash
python src/hal_det/training/train.py --output_dir ./test --num_train_epochs 0.01
```

Resume from checkpoint

```bash
python src/hal_det/training/train.py \
--output_dir ./data/outputs \
--resume_from_checkpoint ./data/outputs/checkpoint-500
```

---

## Dataset

The hallucination detection dataset is available on Hugging Face:

[https://huggingface.co/datasets/pagand/venra](https://huggingface.co/datasets/pagand/venra)

Unlike typical hallucination benchmarks, VeNRA-Data is created using **Adversarial Simulation**, programmatically sabotaging financial records to simulate real system failures such as:

* Numeric Neighbor Traps
* Logic-Code Mismatches
* Temporal Column Drift

---

## Demo

Try the hallucination detection model interactively:

[https://huggingface.co/spaces/pagand/VeNRA_halDet](https://huggingface.co/spaces/pagand/VeNRA_halDet)

---

## Model

LoRA adapter weights:

[https://huggingface.co/pagand/venra](https://huggingface.co/pagand/venra)

---

## Contributing

This project is conducted as part of research in [FactAI Lab](https://gofactai.com). Contributions are welcome in areas such as:

* improving financial table extraction
* expanding adversarial hallucination scenarios
* enhancing trace auditing
* building visualization dashboards for reasoning traces

Please open an issue before submitting major changes.

---

## License

This project is released under:

**Creative Commons Attribution-NonCommercial-ShareAlike 4.0**

[https://creativecommons.org/licenses/by-nc-sa/4.0/](https://creativecommons.org/licenses/by-nc-sa/4.0/)

For enterprise or commercial licensing inquiries:

[pagand@gofactai.com](mailto:pagand@gofactai.com)

---

## Citation

If you use VeNRA in your research, please cite:

```bibtex
@article{agand2026venra,
  title={Neuro-Symbolic Financial Reasoning via Deterministic Fact Ledgers and Adversarial Low-Latency Hallucination Detector},
  author={Agand, Pedram},
  journal={arXiv preprint arXiv:2603.04663},
  year={2026}
}
```
