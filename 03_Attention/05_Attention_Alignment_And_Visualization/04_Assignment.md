# 💡 Attention Alignment and Visualization — Assignment

> Goal: Practice constructing alignment matrices, interpreting heatmaps, reasoning about row/column semantics, identifying reordering and coverage issues, distinguishing sharp vs diffuse attention, and avoiding overclaiming causal explanations.

---

# 🧮 1. Build an Alignment Matrix

Suppose a source sequence has 4 tokens and the decoder produces 3 target tokens.

The attention distributions are:

```text
α_1 = [0.70, 0.20, 0.05, 0.05]

α_2 = [0.10, 0.70, 0.15, 0.05]

α_3 = [0.05, 0.10, 0.20, 0.65]
```

Construct the alignment matrix.

### Answer

Stack one distribution per target timestep:

```text
A =
[
 0.70  0.20  0.05  0.05

 0.10  0.70  0.15  0.05

 0.05  0.10  0.20  0.65
]
```

Therefore:

```text
A ∈ R^(3 × 4)
```

because:

```text
T_tgt = 3
T_src = 4
```

---

# 📐 2. Identify the Matrix Dimensions

Suppose:

```text
T_src = 12
T_tgt = 7
```

What is the attention matrix shape using our convention?

### Answer

```text
A ∈ R^(7 × 12)
```

where:

```text
7 rows
→ target timesteps

12 columns
→ source positions
```

---

# 🧠 3. What Does One Cell Mean?

Suppose:

```text
A[4, 7] = 0.82
```

What does this mean?

### Answer

It means:

> At target timestep 4, the encoder representation at source position 7 receives attention weight 0.82 when constructing the context vector.

It does **not** automatically mean:

> source token 7 causally caused the final prediction.

---

# 🔍 4. Which Axis Is Which?

A heatmap has:

```text
rows = source tokens
columns = target tokens
```

Is the visualization wrong?

### Answer

No.

The plotted orientation is only a convention.

The underlying relation is still:

```text
α_(t,i)
```

between:

```text
target timestep t
```

and:

```text
source position i
```

Always read the axis labels instead of assuming orientation.

---

# 🧮 5. Check Row Normalization

Consider:

```text
A =
[
 0.2  0.3  0.5

 0.1  0.6  0.3

 0.4  0.4  0.2
]
```

Do the rows satisfy the standard attention normalization rule?

### Answer

Yes.

Row 1:

```text
0.2 + 0.3 + 0.5 = 1
```

Row 2:

```text
0.1 + 0.6 + 0.3 = 1
```

Row 3:

```text
0.4 + 0.4 + 0.2 = 1
```

So every row is a valid source-wise attention distribution.

---

# 🧠 6. Do the Columns Need to Sum to 1?

Using:

```text
A =
[
 0.2  0.3  0.5

 0.1  0.6  0.3

 0.4  0.4  0.2
]
```

compute the column sums.

### Answer

Column 1:

```text
0.2 + 0.1 + 0.4 = 0.7
```

Column 2:

```text
0.3 + 0.6 + 0.4 = 1.3
```

Column 3:

```text
0.5 + 0.3 + 0.2 = 1.0
```

This is completely valid.

Only rows are normalized by the standard source-wise softmax.

---

# 🧠 7. Interpret a Diagonal Pattern

Suppose:

```text
        x1    x2    x3

y1     0.9   0.05  0.05

y2     0.05  0.9   0.05

y3     0.05  0.05  0.9
```

What pattern do you see?

### Answer

A strong diagonal.

This suggests approximately:

```text
x1 → y1
x2 → y2
x3 → y3
```

and therefore roughly similar source-target ordering.

---

# 🔄 8. Detect Reordering

Consider:

```text
        A      B      C

A'     0.90   0.05   0.05

C'     0.05   0.10   0.85

B'     0.05   0.85   0.10
```

What does this tell you?

### Answer

The target order is approximately:

```text
A
C
B
```

while the source order is:

```text
A
B
C
```

The attention pattern therefore shows:

> source-target reordering.

This is valid behavior and is one reason attention need not be strictly monotonic.

---

# 🧠 9. Hard or Soft Alignment?

Classify each distribution.

### A

```text
[0, 0, 1, 0]
```

### B

```text
[0.05, 0.15, 0.65, 0.15]
```

### Answer

A:

```text
Hard alignment
```

because only one position is selected.

B:

```text
Soft alignment
```

because relevance is distributed across several positions.

---

# 🧠 10. Phrase-Level Alignment

Suppose the source phrase is:

```text
machine learning
```

and one target token receives:

```text
[0.05, 0.45, 0.45, 0.05]
```

where positions 2 and 3 correspond to:

```text
machine
learning
```

How should you interpret this?

### Answer

The target token is using information from both source representations.

This is a:

> phrase-level / many-to-one soft alignment.

Attention does not require strict word-to-word correspondence.

---

# 🎯 11. Sharp or Diffuse?

Classify the following.

### Distribution A

```text
[0.01, 0.02, 0.94, 0.03]
```

### Distribution B

```text
[0.20, 0.25, 0.30, 0.25]
```

### Answer

A:

```text
sharp / concentrated
```

B:

```text
diffuse / distributed
```

But do not conclude that A is automatically better.

---

# 🧠 12. Is Sharp Attention Always Better?

A candidate says:

> "The sharper the attention, the better the model."

Is this correct?

### Answer

No.

Sharp attention may be appropriate for:

* names
* dates
* numbers
* strongly localized information

Diffuse attention may be appropriate for:

* phrases
* grammar
* sentence-level context
* distributed evidence

The correct concentration depends on the task and decoder step.

---

# 🧮 13. Coverage Calculation

Suppose the matrix is:

```text
A =
[
 0.7  0.2  0.1

 0.6  0.3  0.1

 0.5  0.4  0.1
]
```

Compute:

```text
coverage_i = Σ_t α_(t,i)
```

for each source position.

### Answer

Source 1:

```text
0.7 + 0.6 + 0.5 = 1.8
```

Source 2:

```text
0.2 + 0.3 + 0.4 = 0.9
```

Source 3:

```text
0.1 + 0.1 + 0.1 = 0.3
```

So:

```text
coverage = [1.8, 0.9, 0.3]
```

---

# 🧠 14. Interpret the Coverage

Using:

```text
coverage = [1.8, 0.9, 0.3]
```

what could we infer?

### Answer

Source position 1 receives repeated attention.

Source position 3 receives relatively little attention.

This can motivate investigation into:

* repeated source use
* ignored information
* omissions

But coverage alone does not prove correctness or failure.

---

# 🛡️ 15. Find the Padding Bug

Suppose:

```text
Source:
The price changed today PAD PAD
```

A heatmap shows:

```text
PAD_1
→ 0.18 average attention

PAD_2
→ 0.15 average attention
```

What would you investigate first?

### Answer

The attention mask.

Correct masking should make:

```text
α_PAD ≈ 0
```

because PAD positions should not compete for attention.

---

# 🧠 16. Why Is “Attention to Token i” an Approximation?

A candidate says:

> "Weight 0.9 on word 5 means only word 5 influenced the prediction."

What is wrong with this statement?

### Answer

Attention operates on:

```text
h_5
```

the encoder representation at position 5.

With contextual encoders, especially BiRNNs/BiLSTMs:

```text
h_5
```

can contain information about:

```text
word 5
+
surrounding context
```

So the weight is applied to a contextual representation, not an isolated raw token.

---

# 🧠 17. Contribution or Causation?

Suppose:

```text
α_(t,6) = 0.92
```

Can you say:

> "Source token 6 caused the output token."

### Answer

No.

You can safely say:

> Encoder representation `h_6` received a large coefficient in constructing the attention context for target timestep `t`.

The final output also depends on:

* decoder state
* previous target tokens
* output layer
* other model transformations

So:

```text
high attention
≠
complete causal proof
```

---

# 🔬 18. How Would You Test Causal Importance?

Suppose you suspect source position 6 is crucial.

What stronger experiment could you perform?

### Answer

Possible approaches include:

* remove or mask the source information
* perturb the token or representation
* compare output before and after intervention
* perform an ablation
* construct a counterfactual input

Then observe how the model output changes.

This is stronger evidence than visualization alone.

---

# 🔍 19. Diagnose Repeated Generation

Generated output:

```text
the company the company the company
```

Attention:

```text
t1 → source 4 = 0.82
t2 → source 4 = 0.86
t3 → source 4 = 0.84
t4 → source 4 = 0.88
```

What hypothesis would you form?

### Answer

A reasonable hypothesis is:

> the decoder is repeatedly retrieving the same source region and may have a coverage/repetition problem.

But this is a hypothesis, not proof.

You should also inspect:

* decoder state
* token probabilities
* decoding strategy
* training examples
* coverage
* attention scores

---

# 🔍 20. Diagnose Omitted Information

Source:

```text
Payment of $4.2 million is due Friday.
```

Generated output:

```text
Payment is due Friday.
```

The attention map shows almost no attention on:

```text
$4.2 million
```

What does this suggest?

### Answer

It suggests a possible source-access issue:

```text
important source region
↓
receives little attention
↓
corresponding information omitted
```

But other causes are possible, so inspect:

* encoder representations
* decoder logits
* training data
* decoding
* masking

---

# ✅ True / False

## 1.

One decoder timestep normally produces one distribution over source positions.

**Answer:** True.

---

## 2.

The standard alignment matrix is often written as:

```text
T_tgt × T_src
```

**Answer:** True.

---

## 3.

Every column in the attention matrix must sum to 1.

**Answer:** False.

---

## 4.

Every row normally sums to 1 under source-wise softmax.

**Answer:** True.

---

## 5.

A diagonal heatmap is always correct.

**Answer:** False.

---

## 6.

Off-diagonal patterns can represent valid source-target reordering.

**Answer:** True.

---

## 7.

Soft attention can assign weight to multiple source positions.

**Answer:** True.

---

## 8.

Diffuse attention always indicates confusion.

**Answer:** False.

---

## 9.

Attention on PAD tokens can indicate a masking issue.

**Answer:** True.

---

## 10.

High attention automatically proves causal importance.

**Answer:** False.

---

## 11.

Bahdanau and Luong attention can both produce alignment matrices.

**Answer:** True.

---

## 12.

An attention heatmap alone usually tells you exactly which scoring function generated it.

**Answer:** False.

---

# ⭐ Staff Engineer Challenge

You are debugging a production recurrent summarization system.

For one input:

```text
source length = 300
target length = 40
```

The generated summary:

* repeats one sentence fragment;
* misses an important financial value;
* has correct grammar.

The attention heatmap shows:

* several decoder steps repeatedly focus on the same source region;
* the financial value receives almost no attention;
* PAD positions correctly receive approximately zero attention.

### Questions

1. What hypotheses does the heatmap suggest?
2. Which problem is probably **not** present?
3. Can you conclude that attention is the root cause?
4. What additional evidence would you inspect?
5. What intervention would help test your hypothesis?

---

## Strong Answer

### 1. What hypotheses does the map suggest?

Repeated attention may be related to:

```text
repetition / insufficient coverage
```

while low attention to the financial value may be related to:

```text
information omission / weak source retrieval
```

---

### 2. Which problem is probably not present?

Because:

```text
PAD attention ≈ 0
```

padding masking appears to be working correctly for this example.

---

### 3. Is attention proven to be the root cause?

No.

The heatmap provides evidence and helps form hypotheses.

The output also depends on:

* decoder state
* token history
* output probabilities
* representations
* decoding strategy

---

### 4. What else should be inspected?

I would inspect:

* raw attention scores
* decoder logits
* token probabilities
* beam hypotheses
* encoder representations
* source coverage
* repetition across decoding strategies
* similar successful examples

---

### 5. What intervention could test the hypothesis?

For the omitted financial value:

* perturb or remove the relevant source region;
* compare output behavior;
* inspect whether forcing stronger representation/access changes the result.

For repetition:

* test coverage mechanisms;
* compare greedy vs beam decoding;
* inspect recurrent states;
* intervene on repeated source access.

---

# 🧠 Final Exercise — Reconstruct the Whole Story

Fill in the conceptual pipeline.

```text
Attention at one target step
↓
?
↓
Stack across target steps
↓
?
↓
Visualize
↓
?
↓
Interpret carefully
↓
?
```

### Answer

```text
Attention at one target step
↓
Attention distribution α_t
↓
Stack across target steps
↓
Alignment matrix A
↓
Visualize
↓
Heatmap
↓
Interpret carefully
↓
Diagnostic / alignment insight
```

with the critical warning:

```text
diagnostic signal
≠
complete causal explanation
```

---

# 🎯 Assignment Mental Model

You should now be able to reason through:

```text
One target step
↓
one attention distribution

Many target steps
↓
alignment matrix

Alignment matrix
↓
heatmap

Heatmap
↓
patterns
```

Patterns may include:

```text
diagonal
reordering
sharp
diffuse
repeated
ignored
padding errors
```

But always finish with:

# **Observe → Hypothesize → Test**

not:

# **Observe → Assume causation**

---

# 🔗 Next Topic

We now understand:

```text
how attention is computed
+
how it creates soft source-target alignment
+
how that alignment can be visualized and debugged
```

The final question in this classical attention module is:

> **Even after solving the fixed-context bottleneck, what important problems remain?**

That leads to:

# **06_Attention_Limitations**

and from those limitations, the motivation for:

# **Transformers**
