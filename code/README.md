# Test Code: Reasoning Structure -> Neuron Activation

## Quick Start

```bash
# Install dependencies
pip install transformers torch numpy

# Run all tests
python run_all_tests.py
```

This will:
1. Download GPT-2 Small (~500MB) if not cached
2. Test classifier accuracy on 48 questions
3. Analyze neuron activation patterns
4. Save results to `../results/`

## Files

| File | Description |
|------|-------------|
| `run_all_tests.py` | Main test runner - runs everything |
| `reasoning_classifier.py` | Pattern-based reasoning type classifier |

## What Gets Tested

### 1. Classifier Accuracy
Tests whether linguistic patterns can predict reasoning type:
- Weighting (authority-based)
- Consensus (majority-based)
- Deduction (logic-based)
- Comparison (ordering-based)
- Causal (cause-effect)
- Lookup (retrieval)

### 2. Neuron Activation
For each reasoning type:
- Which neurons fire consistently?
- Which neurons are type-specific?
- How much overlap between types?

### 3. Weighting vs Lookup Deep Dive
Detailed comparison of the two most different reasoning types.

## Expected Output

```
Classifier Accuracy: ~92%
Weighting-Specific Neurons: ~700
Weighting vs Lookup Overlap: ~53%
```

## Using the Classifier

```python
from reasoning_classifier import ReasoningClassifier

classifier = ReasoningClassifier()

result = classifier.classify("Dr. Smith says X. A student disagrees. Who is right?")
print(result.reasoning_type)  # ReasoningType.WEIGHTING
print(result.confidence)      # 1.0
print(result.scores)          # {'weighting': 9.5, 'consensus': 0.0, ...}
```

## Requirements

- Python 3.8+
- transformers
- torch
- numpy

~500MB disk space for GPT-2 model cache.
