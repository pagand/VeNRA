# VeNRA: Verifiable Numerical Reasoning Agent

> **"Trust, but Verify."** — The core philosophy of Financial AI.

VeNRA is an experimental **Financial QA System** designed to solve the "Stochastic Inaccuracy" problem in Large Language Models. It enables users to ask complex numerical questions about 10-K filings (e.g., *"What was the Debt-to-Equity ratio in 2023?"*) and receive answers that are **mathematically deterministic**, **fully traceable**, and **audited in real-time**.

## 🔴 The Problem

Standard RAG (Retrieval-Augmented Generation) pipelines fail in high-stakes financial contexts due to three fundamental limitations:

1.  **The Token Neighbor Problem:** LLMs often hallucinate numbers because "Million" and "Billion" are semantically close in vector space.
2.  **Context Soup:** When retrieving multiple years of data, LLMs struggle to bind specific numbers to specific dates ("Needle in a Haystack" failure).
3.  **Arithmetic Incompetence:** LLMs emulate the *language* of math, not the *logic* of math. They cannot reliably calculate ratios or percentages.

In finance, a 99% accuracy rate results in 0% trust.

## 🟢 The VeNRA Solution

VeNRA moves beyond simple "Text Retrieval" to a **Hybrid Neuro-Symbolic Architecture**. instead of asking the LLM to *guess* the answer, we empower it to *calculate* the answer using verifiable data.

### Key Features

*   **🛡️ Zero-Hallucination Math:** Arithmetic is performed by deterministic code execution, not by predicting the next token.
*   **🔗 Deep Traceability:** Every number in an answer is explicitly linked to a specific row in the financial statements and a specific text chunk in the source PDF.
*   **🤖 The Sentinel (Audit Layer):** A specialized "Judge" model reviews every answer against the source data before it is shown to the user, providing a "Groundedness Score" for confidence.
*   **⚡ Hybrid Retrieval:** Combines semantic search (for concepts) with structured filtering (for precise metrics) to ensure the right data is found every time.

## 🚀 Getting Started

### Prerequisites
*   Python 3.11+
*   API Keys (for reasoning, parsing, monitoring).

### Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/pagand/VeNRA.git
    cd VeNRA
    ```
2.   **Configure:**
    Create a `.env` file in the root directory:
```
GEMINI_API_KEY: Meta analyzer
LLAMA_CLOUD_API_KEY: for parsing
GROQ_API_KEY: for auditor and reasoning generator
OPENROUTER_API_KEY: for auditor and reasoning generator
NVIDIA_API_KEY: for auditor and reasoning generator
HF_TOKEN: for admin access to dataset/spaces/etc.
WANDB_API_KEY: for logging
```

3.  **Set Up Environment:**
We have different dependencies for each phase (1) training and (2) serving. You do not have to install both. 

#### Option 1: Automatic (for training only)

```bash
chmod +x setup.sh
./setup.sh
```

#### Option 2: Manual (for training)

3.1. Create virtual environment
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

3.2. Upgrade pip
```bash
pip install --upgrade pip setuptools wheel
```

3.3. Install PyTorch (for CUDA>13.0)
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

otherwise:
```bash
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

3.4. Verify installation
```bash
python3 << 'EOF'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")
print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
EOF
```

3.5. Install remaining dependencies
```bash
pip install -r requirements_training.txt
```

#### Option 3: Manual (for serving)
3.1. Create virtual environment
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

3.2. Install remaining dependencies
```bash
pip install -r requirements_serving.txt
```

4.  (for training): **Start training**
```bash
python3 src/hal_det/training/train.py
```
4.  (for serving):
**Run the Service**
    ```bash
    uvicorn venra.main:app --reload --app-dir src
    ```
    The Sentinel Service is now active at `http://localhost:8000`.

## 🏗️ Architecture Vision

VeNRA is built on a modular, service-oriented architecture:
1.  **Ingestion Engine:** Converts unstructured PDFs into a structured "Fact Ledger."
2.  **Runtime Agent:** A program-aided agent that generates and executes Python code to answer queries.
3.  **Sentinel Service:** An independent API that acts as the final gatekeeper for quality assurance.

## 🤝 Contributing

This project is an active research prototype. We welcome contributions, especially in:
*   Improving the extraction of complex tables.
*   Enhancing the "Judge" model's ability to detect subtle hallucinations.
*   Building a frontend dashboard for trace visualization.

## 📄 License

Please reach out to us for enterprise and commercial licensing. Contact us at info@upaspro.com.

This project is licensed under the terms of the license included here [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/).
