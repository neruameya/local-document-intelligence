# Local Document Intelligence

A local document intelligence POC using **Qwen2.5-VL 7B**, **GGUF quantization**, and **llama-cpp-python** to classify document images, extract structured information, estimate generation confidence, and route low-confidence results for human review.

The project explores how an open-weight Vision-Language Model (VLM) can be used for document processing without sending document images to an external AI inference API.

---

## 🚀 Overview

Traditional document-processing solutions often combine:

- OCR engines
- Image preprocessing
- Regular expressions
- Rule-based extraction
- Document-specific parsers
- Cloud AI APIs

This project explores an alternative architecture where a multimodal VLM performs document understanding and structured extraction locally.

```text
                    Document Image
                          │
                          ▼
                  Image Encoding
                          │
                          ▼
                  Qwen2.5-VL 7B
                  Local Inference
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
                  Confidence Threshold
                     ┌────┴────┐
                     ▼         ▼
                  Accept    Human Review
```

The objective is not simply to extract information.

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
6. Estimate confidence in the generated response.
7. Route uncertain results for human validation.

Getting an answer from an AI model is relatively easy.

Knowing **when to trust that answer** is a more interesting engineering problem.

---

# 🧠 Model

The current POC uses:

**Qwen2.5-VL 7B Instruct**

The model is a multimodal Vision-Language Model capable of processing:

- Text instructions
- Document images

The model is loaded locally using GGUF artifacts through `llama-cpp-python`.

### Model artifacts

The current setup expects:

```text
Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf
mmproj-qwen-2.5-vl-7B-Instruct-F16.gguf
```

Model weights are **not included in this repository**.

---

# ⚙️ Technology Stack

| Component | Technology |
|---|---|
| Language | Python |
| Vision-Language Model | Qwen2.5-VL 7B Instruct |
| Model Format | GGUF |
| Inference Runtime | llama.cpp |
| Python Binding | llama-cpp-python |
| GPU | NVIDIA 16GB |
| CUDA | CUDA 13.1 |
| Output | Structured JSON |
| Confidence Signal | Token-level log probabilities |
| Runtime | Local / Offline |

---

# 🔐 Why Local Inference?

One of the motivations behind this project is exploring document processing where sensitive documents should remain within an organization's controlled environment.

Local inference can provide:

- Reduced external data exposure
- No dependency on an external inference API
- Greater control over model execution
- Predictable infrastructure costs
- Ability to experiment with open-weight models
- Potential deployment inside controlled enterprise environments

However, **local inference does not automatically make a solution secure**.

A production implementation would still require appropriate:

- Authentication and authorization
- Encryption
- Input validation
- Network controls
- Data retention policies
- Audit logging
- Model governance
- Privacy controls

---

# 🔄 Processing Pipeline

The current implementation follows this flow:

```text
              ┌────────────────────┐
              │   Document Image   │
              └─────────┬──────────┘
                        │
                        ▼
              ┌────────────────────┐
              │ Base64 / Image URI │
              └─────────┬──────────┘
                        │
                        ▼
              ┌────────────────────┐
              │   Qwen2.5-VL 7B   │
              │   Local Inference  │
              └─────────┬──────────┘
                        │
                        ▼
              ┌────────────────────┐
              │  Structured JSON   │
              └─────────┬──────────┘
                        │
                        ▼
              ┌────────────────────┐
              │  Token Logprobs    │
              └─────────┬──────────┘
                        │
                        ▼
              ┌────────────────────┐
              │    Confidence      │
              │    Estimation      │
              └─────────┬──────────┘
                        │
                        ▼
                 ┌──────────────┐
                 │   Threshold  │
                 │     Check    │
                 └──────┬───────┘
                        │
                  ┌─────┴─────┐
                  ▼           ▼
              Accepted    Human Review
```

---

# 📄 Document Understanding

The model receives the document image together with an extraction instruction.

The prompt instructs the model to:

- Classify the document
- Extract required information
- Derive information where required
- Populate the expected JSON structure

This allows the same VLM to perform both:

**Visual understanding + structured extraction**

---

# 🧾 Structured Output

The inference request uses JSON output mode.

Conceptually, a result can look like:

```json
{
  "document_type": "marksheet",
  "student_name": "...",
  "roll_number": "...",
  "subjects": [],
  "total": "...",
  "percentage": "..."
}
```

The actual extraction structure is defined by the project prompt.

Structured output makes the model response easier to validate and integrate with downstream systems.

---

# 📊 Confidence Estimation

A key part of this POC is experimenting with **token-level log probabilities**.

The model provides log probability information for generated tokens.

The implementation converts each log probability into a probability:

```text
probability = exp(log_probability)
```

An average is then calculated across the generated tokens to produce an **overall generation-confidence signal**.

Conceptually:

```text
Token 1 → 0.98
Token 2 → 0.99
Token 3 → 0.94
Token 4 → 0.97
   ...
     │
     ▼
Average token probability
     │
     ▼
Generation confidence
```

## Important distinction

The confidence value should **not** be interpreted as:

> "The model is 96% accurate."

Instead, it represents:

> **Average confidence derived from the model's token probabilities for the generated response.**

Confidence and actual extraction accuracy are different measurements.

A future version of the project will evaluate the relationship between model confidence and actual correctness using a labelled dataset.

---

# 👤 Human-in-the-Loop

The POC uses a configurable confidence threshold.

The current example uses:

```text
confidence threshold = 0.95
```

The decision flow is:

```text
                  Model Output
                       │
                       ▼
               Generation Confidence
                       │
                ┌──────┴──────┐
                │             │
              >= 0.95       < 0.95
                │             │
                ▼             ▼
             Accept       Human Review
```

The purpose is not to eliminate human review.

The purpose is to identify outputs that are more likely to require additional validation.

This provides a basic **human-in-the-loop AI pattern**.

---

# 📈 Initial Result

In the initial test cases, the implementation produced approximately:

> **~96% average generation confidence**

This is an initial POC observation and **not a benchmark**.

It should not be presented as:

- OCR accuracy
- Extraction accuracy
- Field-level accuracy
- Model accuracy

A proper accuracy measurement requires a labelled dataset and comparison against ground truth.

---

# 🧪 Evaluation Roadmap

A production-quality evaluation should measure more than model confidence.

## 1. Extraction Accuracy

```text
Correct fields
──────────────
Total fields
```

## 2. Field-Level Accuracy

Accuracy should eventually be measured independently for fields such as:

- Name
- Roll number
- Subject
- Marks
- Total
- Percentage
- Dates
- Other document-specific fields

## 3. Document Classification Accuracy

```text
Correct classifications
────────────────────────
Total documents
```

## 4. Confidence Calibration

Compare:

```text
Model confidence
       vs
Actual correctness
```

This is particularly important.

A model that is highly confident when it is wrong can be more dangerous than a model that is consistently uncertain.

## 5. Human Review Rate

Measure:

```text
Documents requiring review
──────────────────────────
Total documents
```

The objective is to send the **right documents** to human reviewers rather than simply minimizing the review rate.

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

These metrics provide a foundation for future operational monitoring.

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
                 │      Client / API    │
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
                 │  VLM Inference Layer │
                 │    Qwen2.5-VL 7B    │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Structured Extraction│
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │   Schema Validation  │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │  Confidence Engine   │
                 └──────────┬───────────┘
                            │
                   ┌────────┴────────┐
                   ▼                 ▼
             Auto Accepted       Human Review
                   │                 │
                   └────────┬────────┘
                            ▼
                 ┌──────────────────────┐
                 │  Downstream Systems  │
                 └──────────────────────┘
```

Cross-cutting production concerns include:

- Security
- Access control
- Data protection
- Auditability
- Observability
- Model governance
- Scalability
- Cost management

See [`docs/architecture.md`](docs/architecture.md) for more detail.


---

# 🖥️ Development Environment

The current development environment is:

```text
OS       : Windows 11
GPU      : NVIDIA RTX 
VRAM     : 16 GB
RAM      : 32 GB
CUDA     : 13.1
Python   : 3.14.5
```

`llama-cpp-python` was built locally with GPU support for this environment.

The project therefore provides an opportunity to experiment with the **AI inference layer itself**, rather than only consuming a cloud-hosted AI API.

---

# 📁 Repository Structure

```text
local-document-intelligence/
│
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── .env.example
│
├── src/
│   ├── marksheet-ocr.py
│   └── prompts/
│       └── marksheet_ocr_sys.txt
│
├── input/
│   └── .gitkeep
│
├── output/
│   └── .gitkeep
│
├── examples/
│   └── sample_output.json
│
├── docs/
│   ├── architecture.md
│   ├── evaluation.md
│   
│
└── tests/
```

---

# 🚀 Getting Started

## 1. Clone the repository

```bash
git clone https://github.com/neruameya/local-document-intelligence.git
cd local-document-intelligence
```

## 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

For the local GPU environment, `llama-cpp-python` needs to be installed using a build/wheel compatible with the local Python, CUDA, and GPU environment.

The model weights are not included in this repository.

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
MODEL_ROOT_PATH=D:/models/vision/qwen2.5
```

Adjust the paths for your local environment.

---

# 🧠 Model Setup

Download the appropriate Qwen2.5-VL GGUF artifacts separately.

The current implementation expects:

```text
Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf
mmproj-qwen-2.5-vl-7B-Instruct-F16.gguf
```

Place them in the configured model directory.

**Do not commit model weights to GitHub.**

---

# ▶️ Running the POC

Place supported document images into:

```text
input/
```

Then execute:

```bash
python src/marksheet-ocr.py
```

The application processes the images and generates timestamped JSON results under:

```text
output/
```

A result is conceptually structured like:

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

The values above are illustrative.

---

# 🔒 Security & Privacy

This repository is intended for experimentation and learning.

Do **not** commit:

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

The fact that inference is performed locally does not remove the need for normal application security controls.

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
- Human-review workflow is currently basic.
- No distributed inference architecture.
- No production authentication/authorization layer.
- No persistent audit store.
- No model-serving API layer.

These limitations represent opportunities for future development.

---


# 💡 What This Project Demonstrates

This project is less about building another OCR application and more about exploring the engineering considerations involved in deploying **open-weight multimodal AI**.

It demonstrates concepts around:

- Vision-Language Models
- Local AI inference
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

The broader objective is to understand how modern AI systems can fit into traditional enterprise architecture.

The question being explored is:

> **How can an enterprise architect move from simply consuming AI APIs to understanding and designing the AI inference layer itself?**

This project provides a practical environment for exploring that question using locally hosted open-weight models.

---

# 📚 Documentation

Additional design documentation:

- [`Architecture`](docs/architecture.md)
- [`Evaluation`](docs/evaluation.md)


---

# 📜 Disclaimer

This project is an independent technical exploration.

No proprietary company data, confidential documents, production credentials, or internal implementation details are included in this repository.

All demonstrations should use synthetic or appropriately redacted documents.

---

# 📄 License

This project is licensed under the MIT License.

See [`LICENSE`](LICENSE) for details.