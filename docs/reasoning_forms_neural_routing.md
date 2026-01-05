# Detecting the "Closed Form" of Reasoning: How Question Structure Determines Neural Computation

## Introduction: The Quest for Efficient AI

Large language models are remarkable, but they come with a significant problem: they're expensive to run. Every time you ask ChatGPT a question, thousands of GPUs spin up across data centers, consuming electricity and generating heat, all to produce a response that might be as simple as "Paris" or as complex as a detailed analysis of market trends.

The cost is staggering. OpenAI, Google, and Anthropic collectively spend billions of dollars annually just on inference - the process of running trained models to generate responses. For every dollar spent training a model, companies spend many more dollars actually using it. And as AI becomes more integrated into daily life - powering search engines, writing assistants, coding tools, and customer service bots - these costs only grow.

This has sparked an industry-wide race to make inference more efficient. The approaches are varied:

**Quantization** reduces the precision of numbers in the model. Instead of using 32-bit floating point numbers, you might use 8-bit integers - sacrificing some accuracy for a 4x reduction in memory and computation. It's like using a ruler marked in centimeters instead of millimeters; you lose precision but gain speed.

**Pruning** removes parts of the model that don't contribute much. Like trimming dead branches from a tree, you identify weights that are close to zero and remove them entirely. The model gets smaller and faster, though potentially less capable.

**Distillation** trains a smaller model to mimic a larger one. The "student" model learns to produce similar outputs to the "teacher" model, inheriting much of its capability in a more compact form. It's how we get models that run on phones instead of data centers.

**Mixture of Experts (MoE)** takes a different approach: instead of using the entire model for every query, route different queries to different specialized sub-networks. A question about cooking might activate culinary experts while a math problem activates mathematical experts. The model is large, but only a fraction runs for any given query.

Each approach has tradeoffs. Quantization and pruning risk degrading quality. Distillation requires extensive training. MoE architectures are complex to design and train.

But there's a deeper question lurking beneath all these techniques: **Do we actually need to run the entire model for every query?**

Consider what happens when you ask a simple factual question like "What is the capital of France?" The model processes your question through billions of parameters, activating complex reasoning circuits, weighing possibilities, considering context - all to retrieve a fact that a simple database lookup could provide in microseconds.

Now consider a more complex question: "A Nobel laureate in physics says particles behave one way, while a first-year student disagrees. Who should we believe?" This requires genuine reasoning - weighing authority, considering expertise, understanding the nature of scientific knowledge.

These two questions are fundamentally different. One requires retrieval; the other requires judgment. One is simple; the other is complex. Yet in most language models, both questions flow through the exact same computational machinery.

**What if we could detect what type of thinking a question requires, and route it accordingly?**

This is the question we set out to explore. Not through architectural changes or training modifications, but through analysis: Can we identify the "reasoning fingerprint" of a question? And if so, does that fingerprint correspond to distinct computational pathways within the model?

The implications are significant:

- **For inference providers**: Route simple queries to lightweight pathways, saving compute costs without sacrificing quality on complex queries.

- **For edge deployment**: Run efficient reasoning paths on devices, escalating to full computation only when needed.

- **For interpretability**: Understand what the model is actually doing - which circuits handle which types of reasoning.

- **For reliability**: Detect when a model is using the wrong type of reasoning for a problem, potentially catching errors before they propagate.

What we found surprised us. Not only can we detect reasoning types from surface patterns with over 90% accuracy, but these detected types correspond to genuinely different activation patterns in the model. Nearly half the neurons that fire for one reasoning type don't fire for another.

The "closed form" of reasoning - the minimal operations needed to answer a particular type of question - leaves both linguistic and computational signatures. And those signatures can be detected, measured, and potentially exploited.

---

## Abstract

We discovered that different types of reasoning questions activate different subsets of neurons in GPT-2. By detecting the *reasoning structure* of a question from surface linguistic patterns, we can predict which computational pathways will be used - opening the door to significant inference optimization.

**Key Results:**
- 93.8% accuracy classifying reasoning type from text patterns
- 677 neurons specific to "weighting" reasoning (authority-based)
- Only 53% neuron overlap between weighting and lookup questions
- Potential for 25-47% compute reduction via routing

---

## The Core Insight

When humans answer questions, we don't use the same mental process for every question. Consider:

1. **"What is the capital of France?"** - Simple lookup/retrieval
2. **"Dr. Smith says X, a student says Y. Who's right?"** - Weight by authority
3. **"Nine of ten doctors agree. What's the consensus?"** - Count/majority
4. **"If A then B. A is true. What follows?"** - Logical deduction

These require fundamentally different *reasoning operations*. Our hypothesis: neural networks must also use different computational pathways for different reasoning types.

We tested this on GPT-2 and found **strong evidence that it's true**.

---

## The Six Types of Reasoning

Before diving into results, let's understand what each reasoning type actually means. These aren't arbitrary categories - they represent fundamentally different *operations* that a mind (human or artificial) must perform to arrive at an answer.

### 1. Weighting (Authority-Based Reasoning)

**What it is:** Determining truth by evaluating the credibility or expertise of sources.

**The operation:** Compare the authority/expertise levels of different sources, then trust the more authoritative one.

**Example question:** *"A Nobel laureate in physics says particles behave one way, while a first-year student disagrees. Who should we believe?"*

**How you solve it:** You don't evaluate the physics claim itself - you evaluate *who* is making it. The Nobel laureate has decades of expertise; the student is just learning. Authority wins.

**Linguistic fingerprints:** Expert titles (Dr., Professor), experience markers (veteran, senior), inexperience markers (student, beginner), opinion verbs (says, believes, recommends).

---

### 2. Consensus (Majority-Based Reasoning)

**What it is:** Determining truth by counting how many sources agree.

**The operation:** Count supporters for each position, then go with the majority.

**Example question:** *"Nine out of ten doctors recommend this treatment. One doctor disagrees. What do most doctors recommend?"*

**How you solve it:** You count: 9 vs 1. Nine is more than one. The answer is whatever the nine recommend.

**Linguistic fingerprints:** Quantity words (most, majority, few), fractions (nine out of ten), percentages (90%), agreement words (consensus, agree, support).

**Key difference from weighting:** In weighting, one expert can override a crowd of novices. In consensus, we're counting equally-weighted votes.

---

### 3. Deduction (Logic-Based Reasoning)

**What it is:** Deriving conclusions from premises using logical rules.

**The operation:** Apply logical inference rules (if-then, all-are, none-are) to given facts.

**Example question:** *"All mammals are warm-blooded. A whale is a mammal. Is a whale warm-blooded?"*

**How you solve it:** This is a syllogism. If all X are Y, and Z is an X, then Z must be Y. The answer follows necessarily from the premises - no external knowledge needed.

**Linguistic fingerprints:** Quantifiers (all, every, no, some), conditionals (if, then), logical connectives (therefore, implies, follows), necessity words (must, cannot, necessarily).

**Key insight:** Deduction doesn't require world knowledge - just rule application. The premises contain everything needed.

---

### 4. Comparison (Ordering-Based Reasoning)

**What it is:** Determining relative position in an ordering or ranking.

**The operation:** Build an ordered sequence from pairwise comparisons, then select by position.

**Example question:** *"Mount Everest is taller than K2. K2 is taller than Kangchenjunga. Which is tallest?"*

**How you solve it:** You build the ordering: Everest > K2 > Kangchenjunga. Then you select the extreme: Everest is tallest.

**Linguistic fingerprints:** Comparatives (taller, faster, more expensive), superlatives (tallest, fastest, most), "than" constructions, ordinals (first, last, oldest).

**Key insight:** Comparison requires tracking transitive relationships. If A > B and B > C, then A > C. This is different from simple lookup.

---

### 5. Causal (Cause-Effect Reasoning)

**What it is:** Tracing effects back to causes, or predicting effects from causes.

**The operation:** Follow causal chains to connect observations to explanations.

**Example question:** *"Smoking causes lung cancer. John has lung cancer. What might have contributed?"*

**How you solve it:** You know smoking → lung cancer. John has lung cancer (effect). Trace backwards: smoking is a possible cause.

**Linguistic fingerprints:** Causal verbs (causes, leads to, results in), "because" and "due to", "why" questions, "what caused" phrases.

**Key insight:** Causal reasoning is probabilistic, not certain. Smoking *might* have caused John's cancer - other factors exist. This contrasts with deduction's certainty.

---

### 6. Lookup (Direct Retrieval)

**What it is:** Retrieving a stored fact without transformation.

**The operation:** Match a query to stored knowledge and return the associated value.

**Example question:** *"What is the capital of France?"*

**How you solve it:** You retrieve: France → Paris. No reasoning, no inference, no comparison - just key-value lookup.

**Linguistic fingerprints:** Simple "what is" questions, "who wrote/painted/invented" questions, requests for specific facts (capital of, symbol for, year of).

**Key insight:** Lookup is the simplest operation. It requires memory, not reasoning. A database could answer these questions just as well as an AI.

---

## Why This Matters for Neural Networks

Here's the key insight: **these different operations should require different computations**.

If the brain uses different circuits for remembering facts vs. weighing evidence vs. counting votes, why would an artificial neural network be any different?

Consider what each operation requires:

| Reasoning Type | What Must Be Computed |
|----------------|----------------------|
| Lookup | Content-addressable memory retrieval |
| Weighting | Authority scoring + comparison |
| Consensus | Counting + majority selection |
| Comparison | Ordering + extremum selection |
| Deduction | Rule matching + variable binding |
| Causal | Causal graph traversal |

These are fundamentally different algorithms. The hypothesis: language models have learned to implement these different algorithms in different subsets of their neurons.

If true, we could detect which algorithm a question needs, then route computation accordingly - potentially saving significant compute by skipping irrelevant neural pathways.

---

## What We Found

### 1. Different Reasoning Types Activate Different Neurons

We fed GPT-2 questions requiring six types of reasoning:
- **Weighting**: Authority/expertise determines answer
- **Consensus**: Majority/count determines answer
- **Deduction**: Logical rules determine answer
- **Comparison**: Ordering/ranking determines answer
- **Causal**: Cause-effect determines answer
- **Lookup**: Simple retrieval determines answer

**Result at Layer 6:**

| Reasoning Type | Consistent Neurons | Type-Specific Neurons |
|----------------|-------------------|----------------------|
| Weighting | 1,531 | **143** |
| Consensus | 1,360 | 78 |
| Comparison | 1,307 | 83 |
| Causal | 1,329 | 56 |
| Deduction | 1,181 | 37 |
| Lookup | 865 | 13 |

**Weighting vs Lookup questions share only 52.8% of neurons** - meaning 47% of the computation is different.

### 2. We Can Detect Reasoning Type from Surface Patterns

The "closed form" of reasoning leaves linguistic fingerprints:

**Weighting patterns:**
- Authority markers: "expert", "professor", "Nobel laureate", "Dr."
- Non-authority markers: "student", "beginner", "intern"
- Opinion verbs: "says", "believes", "recommends"
- Comparison questions: "who is more likely correct?"

**Consensus patterns:**
- Majority indicators: "most", "majority", "nine out of ten"
- Agreement words: "consensus", "agree", "support"

**Deduction patterns:**
- Logical connectives: "if", "then", "therefore", "all", "every"
- Inference verbs: "implies", "follows", "conclude"

We built a pattern-based classifier that achieves **93.8% accuracy** at detecting reasoning type.

### 3. Specific Tokens Trigger Reasoning-Specific Neurons

Deep analysis of the 677 weighting-specific neurons revealed their triggers:

| Token | Role | Activation |
|-------|------|------------|
| "says" | Opinion verb | 6.14 |
| "experienced" | Authority marker | 13.22 |
| "first" (as in first-year) | Inexperience marker | 10.65 |
| "new" | Inexperience marker | 9.75 |
| "intern" | Inexperience marker | 8.41 |

The neurons are detecting the **contrast between authority and non-authority** - exactly what weighting reasoning requires.

---

## The "Closed Form" of Reasoning

In mathematics, a closed form is an explicit formula (like `n^2`) rather than a recursive definition. We propose an analogous concept for reasoning:

**The closed form of a reasoning task is the minimal set of operations required to arrive at a valid answer.**

| Reasoning Type | Closed Form |
|----------------|-------------|
| Weighting | `argmax(authority_weight * claim)` |
| Consensus | `argmax(count(supporters))` |
| Deduction | `apply(logical_rules, premises)` |
| Comparison | `sort(items, key) → select(position)` |
| Causal | `trace(effect → cause)` |
| Lookup | `retrieve(key → value)` |

Different closed forms require different computations. The neural network has learned to route to appropriate pathways based on detected reasoning type.

---

## Practical Applications

### 1. Inference Optimization

If we know a question requires lookup reasoning (53% of neurons vs weighting), we could:
- Skip computing the weighting-specific neurons
- Reduce FFN compute by up to 47%
- Use a lightweight classifier as a router

**Architecture:**
```
Input Question
      |
      v
[Reasoning Classifier] -----> Routing Decision
      |                            |
      v                            v
[Type-Specific Neurons]    [Skip Irrelevant Neurons]
      |
      v
    Output
```

### 2. Mixture of Experts Enhancement

Current MoE architectures use learned routers. Our finding suggests:
- Explicit reasoning-type detection could improve routing
- Pattern-based routing is interpretable
- Could combine with learned routing for best of both

### 3. Model Interpretability

Understanding which neurons handle which reasoning types enables:
- Debugging model failures by reasoning type
- Targeted fine-tuning for specific reasoning skills
- Explaining model behavior in human terms

### 4. Efficient Specialized Models

Instead of one large model, deploy:
- Lookup specialist (smallest)
- Deduction specialist
- Weighting specialist
- Route queries to appropriate specialist

---

## How to Reproduce

### Requirements
```
pip install transformers torch numpy
```

### Run All Tests
```bash
cd blog/code
python run_all_tests.py
```

This will:
1. Download GPT-2 Small if not present (~500MB)
2. Run reasoning type neuron analysis
3. Run classifier accuracy tests
4. Generate results in `blog/results/`

### Quick Demo
```python
from reasoning_classifier import ReasoningClassifier

classifier = ReasoningClassifier()

# Weighting question
result = classifier.classify(
    "The expert says X. The novice says Y. Who is correct?"
)
print(result.reasoning_type)  # ReasoningType.WEIGHTING
print(result.confidence)      # 1.0

# Lookup question
result = classifier.classify("What is the capital of France?")
print(result.reasoning_type)  # ReasoningType.LOOKUP
```

---

## Results Summary

| Metric | Value |
|--------|-------|
| Classifier accuracy | 93.8% |
| Weighting-specific neurons | 677 |
| Weighting vs Lookup overlap | 52.8% |
| Potential compute reduction | 25-47% |
| Questions tested | 48 |
| GPT-2 layers analyzed | 12 |

### Per-Type Classifier Accuracy

| Type | Accuracy |
|------|----------|
| Weighting | 100% |
| Deduction | 100% |
| Comparison | 100% |
| Consensus | 88% |
| Causal | 88% |
| Lookup | 88% |

### Neuron Overlap Matrix (Layer 6)

|  | Weighting | Lookup | Deduction |
|--|-----------|--------|-----------|
| Weighting | 100% | 52.8% | 63.8% |
| Lookup | 52.8% | 100% | 63.7% |
| Deduction | 63.8% | 63.7% | 100% |

---

## Limitations and Future Work

### Limitations
1. **Tested on GPT-2 Small only** - larger models may differ
2. **Pattern-based classifier** - may miss edge cases
3. **Static analysis** - didn't measure accuracy impact of routing

### Future Work
1. Test on larger models (GPT-2 Large, LLaMA)
2. Implement actual routing and measure speedup
3. Measure accuracy impact of skipping neurons
4. Learn optimal routing thresholds
5. Combine with quantization/pruning for compound gains

---

## Conclusion

We've shown that:

1. **Reasoning structure is detectable** from surface linguistic patterns
2. **Different reasoning types activate different neurons** in transformers
3. **A simple classifier can predict reasoning type** with 93.8% accuracy
4. **This enables potential compute routing** with 25-47% savings

The "closed form" of reasoning - the minimal operations needed to answer a question type - leaves both linguistic and computational signatures. Detecting these signatures opens new paths for efficient, interpretable AI systems.

---

## Code and Data

- **Test code**: `blog/code/`
- **Results**: `blog/results/`
- **Classifier**: `core/reasoning_classifier.py`

---

*Analysis performed January 2026 on GPT-2 Small (124M parameters)*
