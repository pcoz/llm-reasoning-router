"""
Reasoning Structure Classifier
==============================

Detects the "closed form" of reasoning required by a question.
Based on linguistic patterns that correlate with neuron activation.
"""

from dataclasses import dataclass
from typing import List, Dict
from enum import Enum
import re


class ReasoningType(Enum):
    WEIGHTING = "weighting"
    CONSENSUS = "consensus"
    DEDUCTION = "deduction"
    COMPARISON = "comparison"
    CAUSAL = "causal"
    LOOKUP = "lookup"
    UNKNOWN = "unknown"


@dataclass
class ReasoningClassification:
    """Result of reasoning type classification"""
    reasoning_type: ReasoningType
    confidence: float
    matched_patterns: List[str]
    scores: Dict[str, float]


class ReasoningClassifier:
    """
    Classifies questions by their reasoning structure.
    """

    def __init__(self):
        self.patterns = {
            ReasoningType.WEIGHTING: [
                (r'\b(expert|professor|doctor|specialist|master|senior|chief|lead|experienced|veteran)\b', 2.0),
                (r'\b(Dr\.|PhD|Nobel|laureate|certified|licensed|qualified)\b', 2.0),
                (r'\b(student|beginner|novice|intern|amateur|junior|new|first-year|inexperienced)\b', 1.5),
                (r'\b(says|said|claims|believes|thinks|suggests|recommends|advises|opines)\b', 1.0),
                (r'\b(who.{0,20}(correct|right|trust|reliable|better|more likely))\b', 2.0),
                (r'\b(whose.{0,20}(advice|judgment|opinion|assessment|view))\b', 2.0),
                (r'\b(should (you |we )?(trust|follow|believe|listen to))\b', 1.5),
                (r'(expert|professor|doctor).{0,50}(student|beginner|novice)', 3.0),
                (r'(senior|experienced).{0,50}(junior|new|intern)', 3.0),
            ],
            ReasoningType.CONSENSUS: [
                (r'\b(most|majority|many|several|numerous|widespread)\b', 1.5),
                (r'\b(few|minority|some|rare|uncommon)\b', 1.0),
                (r'\b(\d+\s*(out of|of)\s*\d+)\b', 2.0),
                (r'\b(\d+\s*percent|\d+%)\b', 1.5),
                (r'\b(nine|eight|seven).{0,10}(out of|of).{0,10}ten\b', 2.0),
                (r'\b(agree|consensus|agreement|vote|voted|support)\b', 1.5),
                (r'\b(what (is|was) the (consensus|general|majority))\b', 2.0),
                (r'\b(what do most)\b', 2.0),
            ],
            ReasoningType.DEDUCTION: [
                (r'\b(if|then|therefore|thus|hence|so|because|since)\b', 1.0),
                (r'\b(all|every|no|none|any|some)\b', 1.0),
                (r'\b(implies|follows|conclude|deduce|infer)\b', 2.0),
                (r'\b(must be|cannot be|necessarily|always|never)\b', 1.5),
                (r'(all|every).{0,30}(is|are).{0,30}(is|are)', 2.0),
                (r'if.{0,30}then', 2.0),
                (r'\b(what (can we |do we )?(conclude|infer|deduce))\b', 2.0),
            ],
            ReasoningType.COMPARISON: [
                (r'\b(more|less|greater|smaller|larger|bigger|taller|shorter|faster|slower)\b', 1.0),
                (r'\b(most|least|greatest|smallest|largest|biggest|tallest|fastest)\b', 1.5),
                (r'\b(than)\b', 1.0),
                (r'\b(first|second|third|last|oldest|youngest|newest)\b', 1.0),
                (r'\b(which (is|was|are|were) (the )?(most|least|biggest|smallest))\b', 2.0),
                (r'\b(who (is|was) (oldest|youngest|tallest|fastest))\b', 2.0),
                (r'.{0,30}(than).{0,30}(than)', 2.5),
            ],
            ReasoningType.CAUSAL: [
                (r'\b(causes?|caused|causing)\b', 2.0),
                (r'\b(leads? to|led to|results? in|resulted in)\b', 1.5),
                (r'\b(because|due to|owing to)\b', 1.0),
                (r'\b(why (is|was|did|does|do))\b', 1.5),
                (r'\b(what (caused|might have|could have))\b', 2.0),
                (r'\b(what (is|was) the (reason|cause))\b', 2.0),
            ],
            ReasoningType.LOOKUP: [
                (r'^(what|who|when|where) (is|was|are|were) (the|a) \w+\??$', 3.0),
                (r'\b(capital of|president of|author of|inventor of)\b', 2.5),
                (r'\b(chemical symbol|atomic number)\b', 2.0),
                (r'\b(what year|in what year|when did)\b', 2.0),
                (r'\b(who (wrote|painted|invented|discovered|founded))\b', 2.0),
            ],
        }

    def classify(self, text: str) -> ReasoningClassification:
        text_lower = text.lower()
        scores = {rt: 0.0 for rt in ReasoningType if rt != ReasoningType.UNKNOWN}
        matched = {rt: [] for rt in ReasoningType if rt != ReasoningType.UNKNOWN}

        for reasoning_type, patterns in self.patterns.items():
            for pattern, weight in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    scores[reasoning_type] += weight
                    matched[reasoning_type].append(pattern)

        if all(s == 0 for s in scores.values()):
            return ReasoningClassification(
                reasoning_type=ReasoningType.UNKNOWN,
                confidence=0.0,
                matched_patterns=[],
                scores={k.value: v for k, v in scores.items()}
            )

        best_type = max(scores, key=scores.get)
        total_score = sum(scores.values())
        confidence = scores[best_type] / total_score if total_score > 0 else 0

        return ReasoningClassification(
            reasoning_type=best_type,
            confidence=confidence,
            matched_patterns=matched[best_type],
            scores={k.value: v for k, v in scores.items()}
        )


if __name__ == "__main__":
    classifier = ReasoningClassifier()

    tests = [
        ("Dr. Smith says X. A student disagrees. Who is correct?", "weighting"),
        ("Nine out of ten doctors recommend this. What do most say?", "consensus"),
        ("All mammals are warm-blooded. Is a whale warm-blooded?", "deduction"),
        ("A is taller than B. B is taller than C. Who is tallest?", "comparison"),
        ("Smoking causes cancer. John has cancer. What contributed?", "causal"),
        ("What is the capital of France?", "lookup"),
    ]

    print("Reasoning Classifier Demo")
    print("=" * 50)
    for text, expected in tests:
        result = classifier.classify(text)
        status = "OK" if result.reasoning_type.value == expected else "FAIL"
        print(f"[{status}] {result.reasoning_type.value:12} <- {text[:45]}...")
