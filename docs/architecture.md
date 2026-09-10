# Architecture

## 1. Purpose

This document describes the technical architecture of the Local Document Intelligence POC. The solution uses a locally hosted Vision-Language Model (VLM) to process document images and return structured information. It demonstrates local multimodal inference, structured generation, token-level log probabilities, confidence estimation, and human-review routing.

## 2. High-Level Architecture

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

## 3. Components

### Document Input
The POC processes local document images from the configured input directory.

### Image Encoding
Images are read from disk and encoded as Base64 data URIs for the multimodal request.

### Prompt Layer
A separate system-prompt file defines document classification, extraction, derivation, and the expected JSON structure.

### VLM Inference
The POC uses Qwen2.5-VL-7B-Instruct in GGUF format with its multimodal projection model, loaded through `llama-cpp-python`.

## 4. llama-cpp-python Runtime

The current runtime uses GPU inference and is configured around:

```text
n_ctx = 4096
n_gpu_layers = -1
logits_all = True
```

The `llama-cpp-python` package was locally built for the development environment.

## 5. Structured Generation

The inference request uses JSON output mode. Structured output makes the response easier for downstream systems to consume.

A future production implementation should additionally validate output against an explicit JSON Schema.

## 6. Confidence Calculation

The implementation requests token-level log probabilities and converts them to probabilities:

```text
probability = exp(log_probability)
```

An average across available generated-token probabilities is used as the current overall generation-confidence signal.

**Important:** this is not field-level accuracy and is not OCR accuracy. A confidence of 0.96 does not mean 96% of fields are correct.

## 7. Human Review

The POC uses a configurable threshold, with `0.95` as the current example:

```text
Confidence
    │
    ├── >= threshold ──► Accept
    │
    └── < threshold ───► Human Review
```

Processing errors can also be routed toward review.

## 8. Observability

The POC captures prompt tokens, completion tokens, total tokens, confidence, human-review status, processing duration, input image, and errors.

## 9. Current vs. Target Architecture

### Current POC

```text
Local Image
    │
    ▼
Python Application
    │
    ▼
llama-cpp-python
    │
    ▼
Qwen2.5-VL
    │
    ▼
JSON + Confidence
    │
    ▼
Local Output
```

### Potential Production Architecture

```text
Client
  │
  ▼
API Gateway
  │
  ▼
Document Validation
  │
  ▼
Pre-processing
  │
  ▼
Inference Service
  │
  ▼
Structured Extraction
  │
  ▼
Schema Validation
  │
  ▼
Confidence Engine
  │
  ├───────────────┐
  ▼               ▼
Accepted       Human Review
  │               │
  └───────┬───────┘
          ▼
    Downstream System

Cross-cutting:
Security | Audit | Observability | Governance
```

## 10. Architectural Principles

- **Local-first inference** where appropriate.
- **Structured output** rather than free-form responses.
- **Explicit uncertainty** instead of assuming every response is correct.
- **Human-in-the-loop** for uncertain cases.
- **Observability** for runtime and model behavior.
- **Separation of concerns** between input, prompting, inference, confidence, and output.

## 11. Future Extensions

- REST API
- Async document processing
- Queue-based workload management
- Multiple GPU workers
- Model and prompt versioning
- JSON Schema validation
- Field-level confidence
- Confidence calibration
- Persistent audit logging
- Human-review application
- Authentication and authorization
- Encryption
- Monitoring and alerting
