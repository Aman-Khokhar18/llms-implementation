# Transformer-based Language Models (PyTorch)

This repository is a **portfolio hub** for my implementations of transformer-based language models in **PyTorch**

It groups together three separate projects:

- A **GPT-style** autoregressive language model
- A **LLaMA-style** large language model
- A **vanilla Transformer** encoder–decoder for sequence-to-sequence tasks

---

## 🔧 Tech Stack & Focus

- **PyTorch** – core framework for all model implementations
- **Python** – training & inference scripts
- **Hugging Face Datasets & Tokenizers** (where applicable)
- **ONNX (export-ready)** – models are structured in a way that can be exported to ONNX for deployment
- **TensorRT (optimization-ready)** – ONNX graphs can be further optimized and deployed using NVIDIA TensorRT for low-latency inference on GPUs

> Export & deployment scripts can be added per model (example flow provided below).

---

## 📦 Repositories

### 1. GPT-style Autoregressive Model (PyTorch)

- **Repo:** [GPT\_model](https://github.com/Aman-Khokhar18/GPT_model)
- **Description:** Minimal yet complete GPT-style autoregressive transformer language model in **PyTorch**, trained on WikiText-2 with a custom tokenizer.
- **Highlights:**
  - Causal self-attention with positional embeddings
  - Next-token prediction training loop for language modeling
  - TensorBoard logging for training
  - **Framework:** PyTorch
  - **Deployment:** Architecture suitable for **ONNX export** and **TensorRT** optimization for fast text generation

---

### 2. LLaMA-style Model in PyTorch

- **Repo:** [LLAMA\_model](https://github.com/Aman-Khokhar18/LLAMA_model)
- **Description:** Educational **PyTorch** implementation of the LLaMA transformer architecture, supporting loading official Meta weights and running inference.
- **Highlights:**
  - Pure PyTorch implementation (no external transformer libraries)
  - Rotary positional embeddings, multi-head attention, MLP blocks
  - Support for multiple LLaMA model sizes (e.g., 7B, 13B, 70B)
  - **Framework:** PyTorch
  - **Deployment:** Model structure compatible with **ONNX export** and subsequent **TensorRT** engine creation for production serving

---

### 3. Vanilla Transformer (Encoder–Decoder) in PyTorch

- **Repo:** [Transformer\_model](https://github.com/Aman-Khokhar18/Transformer_model)
- **Description:** From-scratch implementation of the original **Transformer** (from *“Attention is All You Need”*) in **PyTorch**, focused on machine translation.
- **Highlights:**
  - Encoder–decoder Transformer with multi-head attention and positional encoding
  - Trains on Hugging Face’s `opus_books` multilingual dataset
  - Custom bilingual dataset class and tokenizers
  - **Framework:** PyTorch
  - **Deployment:** Design can be exported to **ONNX** (e.g., for translation-as-a-service) and optimized via **TensorRT** on NVIDIA GPUs

---

## 🚀 Export & Deployment (PyTorch → ONNX → TensorRT)

Although the core repositories focus on **PyTorch training and inference**, they are structured so that you can add export/deployment steps like:

### 1. Export a trained PyTorch model to ONNX

Example (GPT model):

```python
import torch
from model import GPTModel  # your model class
from config import get_config

config = get_config()
model = GPTModel(config)
model.load_state_dict(torch.load("weights/best_model.pt", map_location="cpu"))
model.eval()

# Example dummy input (adjust to your actual input shape)
dummy_input = torch.randint(
    low=0,
    high=config["vocab_size"],
    size=(1, config["max_seq_len"]),
    dtype=torch.long,
)

torch.onnx.export(
    model,
    dummy_input,
    "gpt_model.onnx",
    input_names=["input_ids"],
    output_names=["logits"],
    opset_version=17,
    dynamic_axes={"input_ids": {0: "batch", 1: "sequence"}, "logits": {0: "batch", 1: "sequence"}},
)
print("Exported GPT model to gpt_model.onnx")
```

### 2. Optimize the ONNX model with TensorRT

Once exported, you can optimize it with TensorRT, for example using `trtexec`:

```bash
trtexec \
  --onnx=gpt_model.onnx \
  --saveEngine=gpt_model_fp16.engine \
  --fp16 \
  --workspace=4096
```

## 🧠 Skills Demonstrated

Across these projects and this portfolio repo:

- Building **transformer-based language models in PyTorch** (GPT, LLaMA, encoder–decoder)
- Working with **tokenization, datasets, and training pipelines** for NLP
- Structuring models to be **deployment-ready** via **ONNX export** and **TensorRT optimization**
- Writing **clear documentation** and **modular code** suitable for extension and experimentation



