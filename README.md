# Local Document Intelligence

A local document intelligence POC using **Qwen2.5-VL 7B**, **GGUF quantization**, and **llama-cpp-python** to classify documents, extract structured information, estimate generation confidence, and route low-confidence results for human review.

The project explores how an open-weight Vision-Language Model (VLM) can be used for document processing **without sending document images to an external AI API**.

---

## 🚀 Overview

Traditional document-processing pipelines often combine:

- OCR engines
- Image preprocessing
- Rule-based extraction
- Regular expressions
- Document-specific parsers
- Cloud AI APIs

This project explores a different approach:

```text
Document Image
      │
      ▼
Image Encoding
      │
      ▼
Qwen2.5-VL 7B
      │
      ▼
Structured JSON
      │
      ▼
Token Log Probabilities
      │
      ▼
Confidence Estimation
      │
      ▼
 ┌───────────────┐
 │ Confidence    │
 │ Threshold     │
 └───────┬───────┘
         │
    ┌────┴─────┐
    ▼          ▼
 Accept     Human Review
```

The objective is not simply to extract text.

The objective is to explore:

> **How do we know when we should trust the model's output?**

---

# 🎯 Problem Statement

Consider a marksheet or similar semi-structured document.

The system needs to:

1. Identify the document type.
2. Understand the visual structure.
3. Extract relevant fields.
4. Derive information where required.
5. Return the result in a predictable JSON structure.
6. Estimate how confident the model was in its generated output.
7. Route uncertain results for human validation.

This creates an interesting engineering problem.

Getting an answer from an LLM is relatively easy.

**Knowing when to trust that answer is harder.**

---

# 🧠 Model

The current POC uses:

**Qwen2.5-VL 7B Instruct**

The model is used as a multimodal Vision-Language Model capable of processing both:

- Text instructions
- Document images

The model is loaded locally using GGUF artifacts and `llama-cpp-python`.

### Model artifacts

The current setup uses:

```text
Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf
mmproj-qwen-2.5-vl-7B-Instruct-F16.gguf
```

The model files are intentionally **not included in this repository**.

---

# ⚙️ Technology Stack

| Component | Technology |
|---|---|
| Language | Python |
| VLM | Qwen2.5-VL 7B Instruct |
| Model Format | GGUF |
| Inference | llama.cpp |
| Python Binding | llama-cpp-python |
| GPU | NVIDIA 8GB |
| CUDA | CUDA 13.1 |
| Output | Structured JSON |
| Confidence | Token-level log probabilities |
| Runtime | Local / Offline |

---

# 🔐 Why Local Inference?

One of the motivations behind this project is exploring document processing where sensitive documents should not necessarily leave the organization's controlled environment.

A local inference architecture can provide:

- Reduced external data exposure
- No dependency on an external inference API
- Greater control over model execution
- Predictable infrastructure costs
- Ability to experiment with open-weight models
- Potential for deployment inside controlled enterprise environments

This does **not** automatically make the solution secure.

A production implementation would still require appropriate:

- Access control
- Encryption
- Data retention policies
- Audit logging
- Model governance
- Input validation
- Network controls
- Privacy controls

---

# 🔄 Processing Pipeline

The current implementation follows this flow:

```text
             ┌─────────────────┐
             │ Document Image  │
             └────────┬────────┘
                      │
                      ▼
             ┌─────────────────┐
             │ Base64 / Image   │
             │ Data URI         │
             └────────┬────────┘
                      │
                      ▼
             ┌─────────────────┐
             │ Qwen2.5-VL 7B   │
             │ Local Inference  │
             └────────┬────────┘
                      │
                      ▼
             ┌─────────────────┐
             │ Structured JSON │
             └────────┬────────┘
                      │
             ┌────────▼────────┐
             │ Token Logprobs  │
             └────────┬────────┘
                      │
                      ▼
             ┌─────────────────┐
             │ Confidence      │
             │ Estimation      │
             └────────┬────────┘
                      │
                ┌─────▼─────┐
                │ Threshold │
                │   Check   │
                └─────┬─────┘
                      │
              ┌───────┴────────┐
              ▼                ▼
          Accepted         Human Review
```

---

# 📄 Document Understanding

The model receives the document image together with an extraction instruction.

The prompt asks the model to:

- Classify the document
- Extract required information
- Derive information where required
- Populate the expected JSON structure

This allows the same VLM to perform both **visual understanding and structured extraction**.

---

# 🧾 Structured Output

The inference request uses JSON output mode.

Conceptually:

```json
{
  "document_type": "...",
  "student_name": "...",
  "roll_number": "...",
  "subjects": [],
  "total": "...",
  "percentage": "..."
}
```

The actual schema is defined by the project prompt and can be extended for other document types.

Structured output makes the result easier to integrate with downstream systems.

---

# 📊 Confidence Estimation

One of the interesting parts of this POC is the use of **token-level log probabilities**.

The model returns log probability information for generated tokens.

The implementation converts the log probabilities into probabilities:

```text
probability = exp(log_probability)
```

and calculates an average across the generated tokens.

Conceptually:

```text
Token 1 → 0.98
Token 2 → 0.99
Token 3 → 0.94
Token 4 → 0.97
...
             │
             ▼
      Average probability
             │
             ▼
     Generation confidence
```

The current implementation therefore produces an **overall generation-confidence signal**.

### Important distinction

This value should **not** be interpreted as:

> "The model is 96% accurate."

Instead, it represents:

> **Average confidence derived from the model's token probabilities for the generated response.**

This distinction is important because confidence and actual extraction accuracy are different measurements.

---

# 👤 Human-in-the-Loop

The POC uses a configurable confidence threshold.

Current example:

```text
confidence threshold = 0.95
```

Conceptually:

```text
             Model Output
                  │
                  ▼
          Confidence Score
                  │
          ┌───────┴────────┐
          │                │
       >= 0.95          < 0.95
          │                │
          ▼                ▼
       Accept          Human Review
```

This creates a basic human-in-the-loop pattern.

Instead of assuming every model response is correct, the system can identify outputs that require additional validation.

---

# 📈 Initial Result

In the initial test cases, the implementation produced approximately:

> **~96% average generation confidence**

This is an initial POC observation rather than a benchmark.

It should **not** be presented as OCR accuracy or field-level extraction accuracy.

A proper evaluation would require a labelled dataset and comparison against ground truth.

---

# 🧪 Evaluation Roadmap

A production-quality evaluation should measure more than model confidence.

Future evaluation should include:

### Extraction accuracy

```text
Correct fields / Total fields
```

### Field-level accuracy

Measure accuracy independently for:

- Name
- Roll number
- Subject
- Marks
- Total
- Percentage
- Dates
- Other document-specific fields

### Document classification accuracy

```text
Correct classifications / Total documents
```

### Confidence calibration

Compare:

```text
Model confidence
       vs
Actual correctness
```

This is particularly important.

A model that is highly confident when it is wrong can be more dangerous than a model that is consistently uncertain.

### Human-review rate

Measure:

```text
Documents requiring review
/
Total documents
```

The goal is not necessarily to eliminate human review.

The goal is to send **the right documents to human reviewers**.

---

# 📊 Observability

The POC captures execution information including:

- Prompt token count
- Completion token count
- Total token count
- Generation confidence
- Human-review flag
- Processing time
- Image being processed
- Errors encountered

This provides the foundation for future operational monitoring.

A production implementation could additionally capture:

```text
Request ID
Model version
Model hash
Prompt version
Schema version
GPU utilization
Inference latency
Queue time
Retry count
Validation failures
Human-review outcome
Final corrected value
```

---

# 🏗️ Production Architecture

The current implementation is intentionally a POC.

A production architecture could evolve into:

```text
                    ┌──────────────────────┐
                    │     Client / API     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Document Validation  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Image Pre-processing │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ VLM Inference Layer  │
                    │   Qwen2.5-VL 7B     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Structured Extraction│
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Schema Validation    │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Confidence Engine    │
                    └──────────┬───────────┘
                               │
                   ┌───────────┴───────────┐
                   ▼                       ▼
             Auto Accepted             Human Review
                   │                       │
                   └───────────┬───────────┘
                               ▼
                    ┌──────────────────────┐
                    │ Downstream Systems   │
                    └──────────────────────┘
```

Cross-cutting production concerns would include:

```text
Security
Audit
Observability
Model Governance
Access Control
Data Protection
Scalability
Cost Management
```

---

# 🖥️ Hardware

The current development environment was built around:

```text
OS       : Windows 11
GPU      : NVIDIA 
VRAM     : 8 GB
RAM      : 32 GB
CUDA     : 13.1
Python   : 3.14.5
```

`llama-cpp-python` was built locally for this environment with GPU support.

This is useful for experimenting with local inference rather than depending on a cloud-hosted model.

---

# 📁 Repository Structure

```text
local-document-intelligence/
│
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── pyproject.toml
├── .env.example
│
├── src/
│   ├── marksheet_ocr.py
│   └── prompts/
│       └── marksheet_ocr_sys.txt
│
├── input/
│   └── .gitkeep
│
├── output/
│   └── .gitkeep
│
│
├── examples/
│   └── sample_output.json
│
├── docs/
│   ├── architecture.md
│   |
|   ├── evaluation.md
│
└── tests/
    └── .gitkeep
```

---

# 🚀 Getting Started

## 1. Clone the repository

```bash
git clone <your-repository-url>
cd local-document-intelligence
```

---

## 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

---

## 3. Install dependencies

Install the project's Python dependencies:

```bash
pip install -r requirements.txt
```

For the local GPU environment, install the `llama-cpp-python` wheel built for your CUDA/Python environment.

The model itself is **not included in this repository**.

---

## 4. Configure environment variables

Create:

```text
.env
```

based on:

```text
.env.example
```

Example:

```text
PROJECT_ROOT_PATH=D:/workspace/ai/local-document-intelligence
MODEL_ROOT_PATH=D:/models/qwen
```

---

# 🧠 Model Setup

Download the appropriate Qwen2.5-VL GGUF model artifacts separately and place them in your configured model directory.

Expected files:

```text
Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf
mmproj-qwen-2.5-vl-7B-Instruct-F16.gguf
```

Do not commit model weights to GitHub.

---

# ▶️ Running the POC

Place supported document images into:

```text
input/
```

Then execute:

```bash
python src/marksheet_ocr.py
```

The application processes the images and generates a timestamped JSON result under:

```text
output/
```

The result contains information such as:

```json
{
  "image": "...",
  "extracted_data": {},
  "overall_confidence": 0.96,
  "requires_human_review": false,
  "elapsed_seconds": 2.5,
  "token_usage": {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0
  }
}
```

Values shown above are illustrative.

---

# 🔒 Security & Privacy

This repository is intended for experimentation and learning.

Do not commit:

- Real marksheets
- Personal information
- Aadhaar/PAN information
- Student identifiers
- Employer documents
- Proprietary prompts
- Credentials
- API keys
- `.env` files
- Model weights

Use synthetic or appropriately redacted documents for demonstrations.

---

# ⚠️ Current Limitations

This is a **proof of concept**, not a production document-processing platform.

Current limitations include:

- Confidence is based on token probabilities.
- Confidence is not equivalent to extraction accuracy.
- No large labelled evaluation dataset is included.
- No formal confidence calibration has been performed.
- Field-level confidence mapping can be improved.
- Document preprocessing can be expanded.
- Schema validation can be strengthened.
- Human-review workflow is currently conceptual/basic.
- No distributed inference architecture.
- No production authentication/authorization layer.
- No persistent audit store.
- No model-serving API layer.

These limitations are intentional opportunities for future development.

---

# 💡 What This Project Demonstrates

This project is less about building another OCR application and more about exploring the engineering considerations involved in deploying **open-weight multimodal AI**.

It demonstrates concepts around:

- Vision-Language Models
- Local LLM inference
- GGUF quantization
- llama.cpp
- GPU acceleration
- Structured generation
- Token probabilities
- Confidence estimation
- Human-in-the-loop AI
- Observability
- AI system architecture
- Security and privacy considerations
- Productionization of GenAI workloads

---

# 🎓 Learning Objective

The project is also an experiment in understanding how modern AI systems can fit into traditional enterprise architecture.

The broader question is:

> **How can an enterprise architect move from simply consuming AI APIs to understanding and designing the AI inference layer itself?**

This project provides a practical environment for exploring that question using locally hosted open-weight models.

---

# 📜 Disclaimer

This project is an independent technical exploration.

No proprietary company data, confidential documents, production credentials, or internal implementation details are included in this repository.

All demonstrations should use synthetic or appropriately redacted documents.

---

# 📄 License

This project is licensed under the MIT License.

See [LICENSE](LICENSE) for details.
