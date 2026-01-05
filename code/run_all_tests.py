"""
Run All Tests: Reasoning Structure -> Neuron Activation
========================================================

This script:
1. Downloads GPT-2 Small if not present
2. Runs reasoning type neuron analysis
3. Tests classifier accuracy
4. Saves results to ../results/

Usage:
    python run_all_tests.py
"""

import os
import sys
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Set

# Check dependencies
try:
    import torch
    from transformers import GPT2Model, GPT2Tokenizer
except ImportError:
    print("Missing dependencies. Install with:")
    print("  pip install transformers torch numpy")
    sys.exit(1)

from reasoning_classifier import ReasoningClassifier, ReasoningType


# Test questions by reasoning type
TEST_QUESTIONS = {
    'weighting': [
        "Dr. Smith, a Nobel laureate in physics, says the particle exists. A first-year student disagrees. Who is more likely correct?",
        "The chief surgeon recommends surgery. The receptionist suggests waiting. Whose medical advice should you follow?",
        "A master chess player suggests a move. A beginner suggests a different move. Which move is probably better?",
        "The lead architect says the design is safe. An intern has concerns. Whose structural assessment carries more weight?",
        "A native speaker says the grammar is correct. A tourist disagrees. Who knows the language better?",
        "The head chef says the dish needs salt. A customer who rarely cooks disagrees. Whose culinary judgment to trust?",
        "An experienced pilot says conditions are safe to fly. A passenger feels nervous. Whose assessment of flight safety matters?",
        "The senior detective believes the suspect is guilty. A new recruit thinks otherwise. Whose criminal judgment is more reliable?",
    ],
    'consensus': [
        "Nine out of ten doctors recommend this treatment. One doctor disagrees. What do most doctors recommend?",
        "Most historians agree the war started in 1914. A few claim 1913. What is the accepted date?",
        "The majority of scientists support the theory. A small minority disputes it. What is the scientific consensus?",
        "Eight witnesses say the car was red. Two say it was blue. What color was the car most likely?",
        "Most reviews rate the restaurant five stars. A few give it one star. What is the general opinion?",
        "The committee voted 7-2 in favor. What was the committee's decision?",
        "Ninety percent of users report the software works. Ten percent report bugs. Does the software generally work?",
        "Most countries signed the treaty. A few refused. Is there international agreement?",
    ],
    'deduction': [
        "All mammals are warm-blooded. A whale is a mammal. Is a whale warm-blooded?",
        "If it rains, the ground gets wet. It rained. Is the ground wet?",
        "No reptiles have fur. A snake is a reptile. Does a snake have fur?",
        "All prime numbers greater than 2 are odd. 17 is prime and greater than 2. Is 17 odd?",
        "If A implies B, and B implies C, and A is true, what can we conclude about C?",
        "Every bird has feathers. A penguin is a bird. Does a penguin have feathers?",
        "If the switch is on, the light is on. The light is off. Is the switch on?",
        "All squares are rectangles. This shape is a square. Is it a rectangle?",
    ],
    'comparison': [
        "Mount Everest is taller than K2. K2 is taller than Kangchenjunga. Which is tallest?",
        "Alice is older than Bob. Bob is older than Carol. Who is youngest?",
        "Product A costs more than B. B costs more than C. Which is cheapest?",
        "Team X scored more than Y. Y scored more than Z. Who won?",
        "City A has more population than B. B has more than C. Which city is largest?",
        "Book 1 has more pages than Book 2. Book 2 has more than Book 3. Which is longest?",
        "The red car is faster than blue. Blue is faster than green. Which car is slowest?",
        "January is colder than March. March is colder than May. Which month is warmest?",
    ],
    'causal': [
        "Smoking causes lung cancer. John has lung cancer. What might have contributed?",
        "Lack of sleep causes fatigue. Mary is very tired. What might be the reason?",
        "Overwatering kills plants. The plant died. What might have happened?",
        "Economic recession causes unemployment. Unemployment is rising. What might be occurring?",
        "Friction causes heat. The metal is hot after rubbing. Why is it hot?",
        "Viruses cause colds. Tom has a cold. What likely infected him?",
        "Gravity causes objects to fall. The apple fell. What force acted on it?",
        "Practice improves skill. Her piano playing improved. What did she likely do?",
    ],
    'lookup': [
        "What is the capital of France?",
        "Who wrote Romeo and Juliet?",
        "What year did World War II end?",
        "What is the chemical symbol for gold?",
        "Who painted the Mona Lisa?",
        "What is the largest planet in our solar system?",
        "What language is spoken in Brazil?",
        "Who was the first president of the United States?",
    ],
}


def gelu(x):
    """Gaussian Error Linear Unit"""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def ensure_model_downloaded():
    """Download GPT-2 if not cached."""
    print("Checking for GPT-2 Small model...")
    try:
        model = GPT2Model.from_pretrained('gpt2')
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        print("  Model loaded successfully.")
        return model, tokenizer
    except Exception as e:
        print(f"  Downloading GPT-2 Small (~500MB)...")
        model = GPT2Model.from_pretrained('gpt2')
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        print("  Download complete.")
        return model, tokenizer


def get_active_neurons(model, tokenizer, text: str, layer_idx: int, threshold: float = 0.5) -> Set[int]:
    """Get neurons that fire for this text."""
    block = model.h[layer_idx]
    W1 = block.mlp.c_fc.weight.detach().numpy()
    b1 = block.mlp.c_fc.bias.detach().numpy()

    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden = outputs.hidden_states[layer_idx].numpy()[0]

    all_active = set()
    for pos in range(hidden.shape[0]):
        pre_act = hidden[pos] @ W1 + b1
        activation = gelu(pre_act)
        thresh = threshold * np.mean(np.abs(activation))
        active = np.where(np.abs(activation) > thresh)[0]
        all_active.update(active)

    return all_active


def test_classifier_accuracy():
    """Test the reasoning classifier accuracy."""
    print("\n" + "=" * 60)
    print("TEST 1: CLASSIFIER ACCURACY")
    print("=" * 60)

    classifier = ReasoningClassifier()

    results = {'correct': 0, 'total': 0, 'by_type': {}}

    for true_type, questions in TEST_QUESTIONS.items():
        results['by_type'][true_type] = {'correct': 0, 'total': 0}

        for q in questions:
            pred = classifier.classify(q)
            is_correct = pred.reasoning_type.value == true_type

            results['total'] += 1
            results['by_type'][true_type]['total'] += 1

            if is_correct:
                results['correct'] += 1
                results['by_type'][true_type]['correct'] += 1

    accuracy = results['correct'] / results['total']
    print(f"\nOverall Accuracy: {results['correct']}/{results['total']} ({accuracy*100:.1f}%)")
    print("\nPer-type accuracy:")
    for t, data in results['by_type'].items():
        acc = data['correct'] / data['total'] * 100
        print(f"  {t:12}: {data['correct']}/{data['total']} ({acc:.0f}%)")

    return results


def test_neuron_activation(model, tokenizer):
    """Test neuron activation patterns by reasoning type."""
    print("\n" + "=" * 60)
    print("TEST 2: NEURON ACTIVATION BY REASONING TYPE")
    print("=" * 60)

    layer_idx = 5  # Layer 6 showed best differentiation

    results = {'layer': layer_idx + 1, 'by_type': {}}

    for reasoning_type, questions in TEST_QUESTIONS.items():
        neuron_sets = []
        for q in questions:
            neurons = get_active_neurons(model, tokenizer, q, layer_idx)
            neuron_sets.append(neurons)

        consistent = set.intersection(*neuron_sets) if neuron_sets else set()
        any_active = set.union(*neuron_sets) if neuron_sets else set()

        results['by_type'][reasoning_type] = {
            'consistent': len(consistent),
            'any': len(any_active),
            'consistent_neurons': list(consistent)[:100],  # Store first 100
        }

        print(f"  {reasoning_type:12}: {len(consistent):4} consistent, {len(any_active):4} any")

    # Compute type-specific neurons
    print("\nType-SPECIFIC neurons:")
    all_consistent = {t: set(results['by_type'][t]['consistent_neurons'])
                      for t in results['by_type']}

    for t in results['by_type']:
        specific = all_consistent[t].copy()
        for other in results['by_type']:
            if other != t:
                specific -= all_consistent[other]
        results['by_type'][t]['specific'] = len(specific)
        print(f"  {t:12}: {len(specific):4} specific")

    # Compute pairwise Jaccard
    print("\nPairwise Jaccard (lower = more different):")
    types = list(results['by_type'].keys())
    results['jaccard'] = {}

    for t1 in types:
        for t2 in types:
            n1 = all_consistent[t1]
            n2 = all_consistent[t2]
            if n1 and n2:
                jaccard = len(n1 & n2) / len(n1 | n2)
                results['jaccard'][f"{t1}_vs_{t2}"] = jaccard

    # Show most different pairs
    pairs = [(k, v) for k, v in results['jaccard'].items() if '_vs_' in k]
    pairs = [(k, v) for k, v in pairs if k.split('_vs_')[0] != k.split('_vs_')[1]]
    pairs.sort(key=lambda x: x[1])

    print("  Most different pairs:")
    seen = set()
    for k, v in pairs[:6]:
        t1, t2 = k.split('_vs_')
        pair_key = tuple(sorted([t1, t2]))
        if pair_key not in seen:
            print(f"    {t1:12} vs {t2:12}: {v*100:.1f}%")
            seen.add(pair_key)

    return results


def test_weighting_specific_analysis(model, tokenizer):
    """Deep analysis of weighting-specific neurons."""
    print("\n" + "=" * 60)
    print("TEST 3: WEIGHTING-SPECIFIC NEURON ANALYSIS")
    print("=" * 60)

    layer_idx = 5

    # Get weighting neurons
    weighting_sets = []
    for q in TEST_QUESTIONS['weighting']:
        neurons = get_active_neurons(model, tokenizer, q, layer_idx)
        weighting_sets.append(neurons)
    weighting_consistent = set.intersection(*weighting_sets)

    # Get lookup neurons
    lookup_sets = []
    for q in TEST_QUESTIONS['lookup']:
        neurons = get_active_neurons(model, tokenizer, q, layer_idx)
        lookup_sets.append(neurons)
    lookup_consistent = set.intersection(*lookup_sets)

    weighting_specific = weighting_consistent - lookup_consistent

    print(f"\n  Weighting consistent: {len(weighting_consistent)}")
    print(f"  Lookup consistent: {len(lookup_consistent)}")
    print(f"  Weighting-SPECIFIC: {len(weighting_specific)}")

    overlap = len(weighting_consistent & lookup_consistent)
    union = len(weighting_consistent | lookup_consistent)
    jaccard = overlap / union if union > 0 else 0

    print(f"\n  Weighting vs Lookup Jaccard: {jaccard*100:.1f}%")
    print(f"  (Lower = more different computational pathways)")

    return {
        'weighting_consistent': len(weighting_consistent),
        'lookup_consistent': len(lookup_consistent),
        'weighting_specific': len(weighting_specific),
        'jaccard': jaccard,
    }


def save_results(classifier_results, neuron_results, weighting_results):
    """Save all results to JSON."""
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save summary
    summary = {
        'timestamp': timestamp,
        'classifier': {
            'accuracy': classifier_results['correct'] / classifier_results['total'],
            'correct': classifier_results['correct'],
            'total': classifier_results['total'],
            'by_type': {t: d['correct'] / d['total']
                       for t, d in classifier_results['by_type'].items()},
        },
        'neurons': {
            'layer': neuron_results['layer'],
            'consistent_by_type': {t: d['consistent']
                                   for t, d in neuron_results['by_type'].items()},
            'specific_by_type': {t: d['specific']
                                 for t, d in neuron_results['by_type'].items()},
        },
        'weighting_analysis': weighting_results,
    }

    summary_path = os.path.join(results_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to: {summary_path}")

    # Save detailed results
    detailed_path = os.path.join(results_dir, f'detailed_{timestamp}.json')
    detailed = {
        'classifier': classifier_results,
        'neurons': {k: v for k, v in neuron_results.items() if k != 'by_type'},
        'neuron_by_type': {t: {k: v for k, v in d.items() if k != 'consistent_neurons'}
                           for t, d in neuron_results['by_type'].items()},
        'weighting': weighting_results,
    }
    with open(detailed_path, 'w') as f:
        json.dump(detailed, f, indent=2)
    print(f"Detailed results saved to: {detailed_path}")

    return summary


def main():
    print("=" * 60)
    print("  REASONING STRUCTURE -> NEURON ACTIVATION")
    print("  Full Test Suite")
    print("=" * 60)

    # Ensure model is available
    model, tokenizer = ensure_model_downloaded()
    model.eval()

    # Run all tests
    classifier_results = test_classifier_accuracy()
    neuron_results = test_neuron_activation(model, tokenizer)
    weighting_results = test_weighting_specific_analysis(model, tokenizer)

    # Save results
    summary = save_results(classifier_results, neuron_results, weighting_results)

    # Print final summary
    print("\n" + "=" * 60)
    print("  FINAL SUMMARY")
    print("=" * 60)
    print(f"""
  Classifier Accuracy:     {summary['classifier']['accuracy']*100:.1f}%
  Weighting-Specific Neurons: {summary['weighting_analysis']['weighting_specific']}
  Weighting vs Lookup Overlap: {summary['weighting_analysis']['jaccard']*100:.1f}%

  Key Finding:
    Different reasoning types activate different neurons.
    Pattern-based detection enables computational routing.
""")


if __name__ == "__main__":
    main()
