# Evaluation

## 1. Purpose

Evaluation determines whether the document intelligence system actually produces correct results.

The key distinction is:

> **Generation confidence is not the same as extraction accuracy.**

The current POC observed approximately **~96% average generation confidence** on initial test cases. This must not be interpreted as 96% OCR or field-extraction accuracy.

## 2. Evaluation Dimensions

A useful evaluation should measure:

1. Document classification accuracy
2. Field extraction accuracy
3. Confidence quality/calibration
4. Human-review effectiveness
5. Runtime and operational metrics

## 3. Ground-Truth Dataset

Create a labelled dataset with a ground-truth JSON record for each document.

```text
documents/
├── sample_001.png
├── sample_002.png
└── ...

ground_truth/
├── sample_001.json
├── sample_002.json
└── ...
```

Use synthetic or appropriately redacted documents.

## 4. Document Classification Accuracy

```text
Classification Accuracy
=
Correct Classifications
/
Total Documents
```

## 5. Field-Level Extraction Accuracy

Compare every extracted field with its corresponding ground-truth value.

```text
Field Accuracy
=
Correct Field Values
/
Total Expected Field Values
```

Potential fields include name, roll number, subject marks, total, percentage, and dates.

## 6. Exact vs. Normalized Matching

Comparison rules should depend on the field.

| Field | Comparison |
|---|---|
| Name | Normalized string |
| Roll number | Exact/normalized string |
| Marks | Numeric |
| Percentage | Numeric with tolerance |
| Date | Normalized date |

## 7. Confidence Evaluation

For every prediction, compare model confidence with actual correctness:

```text
Model Confidence
       │
       ▼
Prediction
       │
       ▼
Ground Truth
       │
       ▼
Correct / Incorrect
```

The important question is: when the model reports high confidence, how often is it actually correct?

## 8. Confidence Calibration

Group predictions into confidence ranges and compare them with empirical accuracy.

| Confidence | Predictions | Correct | Empirical Accuracy |
|---|---:|---:|---:|
| 0.90–0.95 | 100 | 82 | 82% |
| 0.95–0.97 | 100 | 91 | 91% |
| 0.97–1.00 | 100 | 95 | 95% |

The desired behavior is generally increasing empirical accuracy with increasing confidence.

## 9. Human-Review Threshold

The current example threshold is `0.95`.

Threshold selection should be based on evaluation data and the business cost of incorrect acceptance versus unnecessary human review.

## 10. False Acceptance vs. False Review

### False acceptance
The system automatically accepts an incorrect result.

### False review
The system sends a correct result to a human reviewer.

False acceptance can be substantially more serious depending on the business use case.

## 11. Human Review Rate

```text
Human Review Rate
=
Documents Sent for Review
/
Total Documents
```

The objective is not simply to minimize review rate, but to find a useful accuracy/automation balance.

## 12. Operational Metrics

Measure:

```text
Documents processed
Average latency
P95 latency
P99 latency
Prompt tokens
Completion tokens
Total tokens
GPU utilization
Human-review rate
Error rate
Retry rate
```

## 13. Dataset Design

The evaluation set should include variation such as:

- Different layouts
- Different resolutions
- Different image qualities
- Different fonts
- Different templates
- Rotated documents
- Blurred documents
- Low-light images
- Missing fields
- Unexpected formatting

## 14. Evaluation Report

A future report should contain:

```text
Dataset
--------
Number of documents
Document types
Number of fields

Accuracy
--------
Classification accuracy
Field accuracy
Document exact-match accuracy

Confidence
----------
Average confidence
Confidence distribution
Calibration analysis

Human Review
------------
Review rate
Correct-review rate
Incorrect-acceptance rate

Performance
-----------
Average latency
P95 latency
Tokens/document
GPU utilization

Errors
------
Common extraction failures
Common classification failures
Low-confidence cases
```

## 15. Current Status

Implemented:

- Token-level log probability collection
- Average generation-confidence calculation
- Confidence threshold
- Human-review routing
- Token usage measurement
- Processing-time measurement

Not yet established:

- Large labelled benchmark
- Ground-truth extraction accuracy
- Field-level accuracy
- Confidence calibration
- Statistical significance
- Production error rates

Therefore:

> **~96% should currently be described as average generation confidence from initial test cases, not model accuracy.**

## 16. Next Milestone

Build a small labelled dataset and measure:

```text
Confidence  ─────────► Actual Correctness
```

This will establish whether the current confidence signal is useful for deciding when to trust the model.
