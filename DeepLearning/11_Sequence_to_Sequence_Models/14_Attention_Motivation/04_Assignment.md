# 💡 Assignment — Attention Motivation

> **Goal:** Apply the motivation behind attention by diagnosing fixed-context failures, distinguishing capacity from accessibility, reasoning about dynamic source access, interpreting simple attention weights, and separating attention from Beam Search, Teacher Forcing, LSTM, and vanishing-gradient problems.

---

# 🧩 Task 1 — Identify the Bottleneck

A classic Seq2Seq model encodes:

```text
x1, x2, x3, ..., x20
```

into:

```text
c = h20
```

The decoder uses the same `c` for every output step.

What is the main architectural weakness?

### ✅ Answer

The model has a:

# **fixed context-vector bottleneck**

All source information must pass through:

> one fixed-size representation.

---

# 🧠 Task 2 — Capacity or Accessibility?

Suppose the encoder context dimension is increased:

```text
512 → 2048
```

The model improves slightly, but still drops names from long sentences.

Which problem may remain?

### ✅ Answer

# **Accessibility**

A larger vector gives:

> more capacity,

but the decoder still has to retrieve all source details from:

> one compressed representation.

So:

```text
Bigger Hidden Size
→ more capacity

Attention
→ better accessibility
```

---

# 🔍 Task 3 — Which Source Information Matters?

Source:

```text
I sent the blue document to Maria yesterday
```

Imagine the decoder is generating a target token corresponding to:

```text
Maria
```

Which source information should ideally receive more emphasis?

### ✅ Answer

The encoder representation associated with:

# `Maria`

should receive higher relevance.

The key idea is:

> different target steps need different source information.

---

# 🔄 Task 4 — Fixed vs Dynamic Context

Fill the blanks.

Classic Seq2Seq:

```text
Decoder step 1 → ______
Decoder step 2 → ______
Decoder step 3 → ______
```

Attention-based Seq2Seq:

```text
Decoder step 1 → ______
Decoder step 2 → ______
Decoder step 3 → ______
```

### ✅ Answer

Classic:

```text
Decoder step 1 → c
Decoder step 2 → c
Decoder step 3 → c
```

Attention:

```text
Decoder step 1 → c1
Decoder step 2 → c2
Decoder step 3 → c3
```

The attention context can change at every target step.

---

# 📐 Task 5 — Interpret the Attention Equation

Given:

# `c_t = Σ_i α_(t,i) h_i`

explain each term.

### ✅ Answer

* `h_i` = source/encoder representation at position `i`
* `α_(t,i)` = relevance weight of source position `i` for decoder step `t`
* `c_t` = weighted source context used at decoder step `t`

---

# 🧮 Task 6 — Calculate a Context Vector

Suppose:

```text
h1 = [1, 0]
h2 = [0, 2]
h3 = [2, 2]
```

and at decoder step `t`:

```text
α1 = 0.2
α2 = 0.3
α3 = 0.5
```

Calculate:

`c_t`

### ✅ Answer

`c_t = 0.2h1 + 0.3h2 + 0.5h3`

So:

```text
0.2[1,0] = [0.2, 0]
0.3[0,2] = [0, 0.6]
0.5[2,2] = [1.0, 1.0]
```

Add:

# `c_t = [1.2, 1.6]`

---

# 🧠 Task 7 — Interpret Attention Weights

Suppose:

```text
α_t = [0.05, 0.10, 0.75, 0.10]
```

What does this tell you?

### ✅ Answer

The decoder step focuses most strongly on:

# **source position 3**

because it has weight:

`0.75`

But the context still includes information from the other source positions.

This is:

# **soft attention**

---

# 🚨 Task 8 — Is This Hard Alignment?

A developer says:

> “Because source position 3 has the largest weight, attention selects only position 3.”

Correct?

### ✅ Answer

No.

If the weights are:

```text
0.05, 0.10, 0.75, 0.10
```

the context includes:

> all four states,

just with different strengths.

Attention is usually:

> weighted soft alignment,

not exact hard selection.

---

# 📐 Task 9 — Weight Sanity Check

Which set looks like a valid normalized attention distribution?

### A

```text
[0.2, 0.5, 0.3]
```

### B

```text
[0.7, 0.7, -0.4]
```

### ✅ Answer

# A

because:

```text
0.2 + 0.5 + 0.3 = 1
```

and all values are non-negative.

Typical attention weights satisfy:

`α_i ≥ 0`

and:

`Σ_i α_i = 1`

---

# 🔗 Task 10 — Match the Problem to the Technique

Match:

### A. LSTM

### B. Attention

### C. Beam Search

### D. Teacher Forcing

with:

### 1. Target sequence search

### 2. Better recurrent memory

### 3. Training target-history strategy

### 4. Dynamic source access

### ✅ Answer

```text
A → 2
B → 4
C → 1
D → 3
```

---

# 🧠 Task 11 — Diagnose the Wrong Fix

A translation model drops information from early source positions.

A developer proposes:

> “Increase beam size from 5 to 20.”

Why may this fail to solve the root problem?

### ✅ Answer

Beam Search improves:

> target-side search.

But if source information has already been:

> compressed poorly or made inaccessible,

searching more target hypotheses cannot restore that information.

The likely issue is:

> source access,

not target search.

---

# 🚨 Task 12 — Attention vs Vanishing Gradient

A developer says:

> “Attention was invented because gradients vanish in RNNs.”

What is missing from this explanation?

### ✅ Answer

The main motivation in this Seq2Seq story is:

# **fixed source-context bottleneck and source accessibility**

Vanishing gradients are:

> an optimization/gradient-flow issue.

Attention can create shorter information paths, but it should not be reduced to:

> a vanishing-gradient solution.

---

# 🔍 Task 13 — Why Preserve All Encoder States?

Classic model retains only:

```text
h_T
```

Attention model retains:

```text
h1, h2, ..., hT
```

What practical advantage does this provide?

### ✅ Answer

The decoder can revisit:

> different source positions

when different information becomes relevant.

This helps preserve:

* names
* numbers
* dates
* early clauses
* long-range source details

---

# 🧠 Task 14 — Alignment Is Not One-to-One

Source phrase:

```text
machine learning
```

might map to a multi-token target phrase.

Does this violate the idea of attention alignment?

### ✅ Answer

No.

Attention alignment is:

> soft relevance,

not necessarily:

> one source token ↔ one target token.

One target token can attend to several source states, and multiple target tokens may attend to the same source region.

---

# 🔄 Task 15 — Trace the Conceptual Attention Pipeline

Put these in order:

* Weighted source context
* Relevance scores
* Encoder states
* Attention weights
* Decoder need/state

### ✅ Answer

```text
Decoder Need / State
+
Encoder States
↓
Relevance Scores
↓
Attention Weights
↓
Weighted Source Context c_t
```

---

# 🧠 Task 16 — Diagnose a Long-Sequence Failure

A model works well on 5-word inputs but poorly on 50-word inputs.

Symptoms:

* names disappear
* numbers change
* early information is forgotten
* final source words dominate

What architectural limitation does this strongly suggest?

### ✅ Answer

A likely issue is:

# **fixed-context compression / accessibility bottleneck**

As source information grows:

```text
Source Information ↑
while
Fixed Interface stays same
```

Attention directly targets this limitation.

---

# ⭐ Task 17 — Dynamic Retrieval Mental Model

Complete:

```text
Encoder States
→ __________

Attention Weights
→ __________

c_t
→ __________
```

### ✅ Answer

```text
Encoder States
→ stored source representations

Attention Weights
→ relevance

c_t
→ dynamically retrieved source context
```

---

# ✅ True / False

| #  | Statement                                                                  | Answer  |
| -- | -------------------------------------------------------------------------- | ------- |
| 1  | Classic Seq2Seq may use one fixed context for all decoder steps.           | ✅ True  |
| 2  | Increasing hidden size and using attention solve exactly the same problem. | ❌ False |
| 3  | Attention gives each decoder step dynamic source access.                   | ✅ True  |
| 4  | `c_t` must be identical for every target step.                             | ❌ False |
| 5  | Attention weights typically represent relative source relevance.           | ✅ True  |
| 6  | Soft attention must select exactly one source state.                       | ❌ False |
| 7  | Source-target alignment must always be one-to-one.                         | ❌ False |
| 8  | Attention and Beam Search solve different problems.                        | ✅ True  |
| 9  | Early attention systems could still use LSTMs.                             | ✅ True  |
| 10 | Attention was only a vanishing-gradient fix.                               | ❌ False |
| 11 | Keeping all encoder states reduces fixed-context pressure.                 | ✅ True  |
| 12 | Attention can provide direct access to early source states.                | ✅ True  |

---

# ⭐ Staff Engineer Challenge — Long-Document Translation Failure

## Scenario

A production translation model uses:

```text
BiLSTM Encoder
↓
Final Encoder Representation
↓
LSTM Decoder
↓
Beam Search
```

Short inputs perform well.

Long inputs show:

* dropped named entities
* missing dates
* early clauses ignored
* later source content translated more reliably
* beam size increases give little improvement
* doubling hidden size improves quality only slightly

---

## Question 1 — What is the most likely architectural bottleneck?

### ✅ Answer

The fixed encoder-to-decoder communication interface.

Even with a BiLSTM, the decoder relies on:

> a fixed summary of a variable-length source.

This creates both:

* capacity pressure
* accessibility pressure

---

## Question 2 — Why did a larger hidden dimension only partially help?

### ✅ Answer

It increased:

> representational capacity.

But did not change:

> how the decoder retrieves particular source details.

The decoder still depends on one global compressed representation.

---

## Question 3 — Why did wider Beam Search provide little benefit?

### ✅ Answer

Because Beam Search improves:

> target hypothesis search.

If the decoder cannot access the correct source detail, wider search cannot fully compensate.

This suggests:

> model/source-access limitation rather than primarily search error.

---

## Question 4 — What architectural experiment would you run?

### ✅ Answer

Retain:

```text
h1, h2, ..., hT
```

and add dynamic source access so that:

# `c_t = Σ_i α_(t,i)h_i`

for each target step.

This allows target steps to retrieve different source information.

---

## Question 5 — What behavior would indicate the experiment worked?

### ✅ Answer

I would expect improvements in:

* named-entity retention
* number/date accuracy
* early-source detail preservation
* translation quality vs source length
* alignment between relevant source and target positions

---

## Question 6 — What new cost would you monitor?

### ✅ Answer

Attention requires comparing decoder needs against multiple source states.

So I would monitor:

* inference latency
* memory
* compute cost
* cost growth with source length and target length

Conceptually:

# roughly `T × T'` source-target relevance work

for recurrent encoder-decoder attention.

---

## Question 7 — Would attention make Beam Search unnecessary?

### ✅ Answer

No.

They solve different problems.

```text
Attention
→ what source information should I use?

Beam Search
→ what target sequence should I keep?
```

A system can use:

> both attention and Beam Search.

---

# 🎯 Final Applied Exercise

Complete the architectural evolution:

```text
Source Sequence
↓
Encoder
↓
Classic Model uses __________
↓
Problem: __________
↓
Keep __________
↓
At each decoder step compute __________
↓
Create __________
↓
Decoder gets __________ source information
```

### ✅ Answer

```text
Source Sequence
↓
Encoder
↓
Classic Model uses ONE FIXED CONTEXT
↓
Problem: CONTEXT BOTTLENECK / POOR ACCESSIBILITY
↓
Keep ALL ENCODER STATES
↓
At each decoder step compute RELEVANCE WEIGHTS
↓
Create DYNAMIC CONTEXT c_t
↓
Decoder gets STEP-SPECIFIC source information
```

---

# 🧠 Final Assignment Mental Model

Before attention:

```text
Variable-Length Source
↓
Compress Everything
↓
ONE c
↓
Use Same c for Every Target Step
```

Problem:

```text
More Source Information
↓
More Compression Pressure
↓
Relevant Details Harder to Retrieve
```

Attention changes this to:

```text
h1 h2 h3 ... hT
       │
       ▼
Current Decoder Need
       │
       ▼
Relevance
       │
       ▼
α_(t,i)
       │
       ▼
c_t = Σ_i α_(t,i)h_i
       │
       ▼
Current Target Prediction
```

The deepest shift is:

# **Compress Everything Once → Retrieve What Matters Now**

---

# ⭐ Golden Rule

> **When diagnosing why attention is needed, do not ask only whether the encoder can store enough information. Ask whether the decoder can access the right source information at the exact moment it needs it. Attention was the architectural answer to that accessibility problem.**
